#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16 // comment out this if want to use FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

// Creat the engine using only the API and not any parser.
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, float& gd, float& gw)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov4-p6.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // yolov4-p6 backbone
    auto conv0 = convBlock(network, weightMap, *data, get_width(32, gw), 3, 1, 1, "model.0");
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(64, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), get_width(64, gw), get_width(64, gw), get_depth(1, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(128, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(256, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(15, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(512, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = bottleneckCSP(network, weightMap, *conv7->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(15, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto bottleneck_csp10 = bottleneckCSP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(7, gd), true, 1, 0.5, "model.10");
    auto conv11 = convBlock(network, weightMap, *bottleneck_csp10->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.11");
    auto bottleneck_csp12 = bottleneckCSP(network, weightMap, *conv11->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(7, gd), true, 1, 0.5, "model.12");

    // yolov4-p6 head
    auto sppcsp13 = SPPCSP(network, weightMap, *bottleneck_csp12->getOutput(0), get_width(512, gw), get_width(512, gw), 0.5, 5, 9, 13, "model.13");
    auto conv14 = convBlock(network, weightMap, *sppcsp13->getOutput(0), get_width(512, gw), 1, 1, 1, "model.14");
    auto deconv15 = upSample(network, weightMap, *conv14->getOutput(0), get_width(512, gw));
    auto conv16 = convBlock(network, weightMap, *bottleneck_csp10->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    ITensor *inputTensors17[] = {conv16->getOutput(0), deconv15->getOutput(0)};
    auto cat17 = network->addConcatenation(inputTensors17, 2);
    auto bottleneck_csp18 = bottleneckCSP2(network, weightMap, *cat17->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.18");
    auto conv19 = convBlock(network, weightMap, *bottleneck_csp18->getOutput(0), get_width(256, gw), 1, 1, 1, "model.19");
    auto deconv20 = upSample(network, weightMap, *conv19->getOutput(0), get_width(256, gw));
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(256, gw), 1, 1, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), deconv20->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP2(network, weightMap, *cat22->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    auto conv24 = convBlock(network, weightMap, *bottleneck_csp23->getOutput(0), get_width(128, gw), 1, 1, 1, "model.24");
    auto deconv25 = upSample(network, weightMap, *conv24->getOutput(0), get_width(128, gw));
    auto conv26 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(128, gw), 1, 1, 1, "model.26");
    ITensor *inputTensors27[] = {conv26->getOutput(0), deconv25->getOutput(0)};
    auto cat27 = network->addConcatenation(inputTensors27, 2);
    auto bottleneck_csp28 = bottleneckCSP2(network, weightMap, *cat27->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), false, 1, 0.5, "model.28");
    auto conv29 = convBlock(network, weightMap, *bottleneck_csp28->getOutput(0), get_width(256, gw), 3, 1, 1, "model.29");

    auto conv30 = convBlock(network, weightMap, *bottleneck_csp28->getOutput(0), get_width(256, gw), 3, 2, 1, "model.30");
    ITensor *inputTensors31[] = {conv30->getOutput(0), bottleneck_csp23->getOutput(0)};
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto bottleneck_csp32 = bottleneckCSP2(network, weightMap, *cat31->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.32");
    auto conv33 = convBlock(network, weightMap, *bottleneck_csp32->getOutput(0), get_width(512, gw), 3, 1, 1, "model.33");

    auto conv34 = convBlock(network, weightMap, *bottleneck_csp32->getOutput(0), get_width(512, gw), 3, 2, 1, "model.34");
    ITensor *inputTensors35[] = {conv34->getOutput(0), bottleneck_csp18->getOutput(0)};
    auto cat35 = network->addConcatenation(inputTensors35, 2);
    auto bottleneck_csp36 = bottleneckCSP2(network, weightMap, *cat35->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.36");
    auto conv37 = convBlock(network, weightMap, *bottleneck_csp36->getOutput(0), get_width(1024, gw), 3, 1, 1, "model.37");

    auto conv38 = convBlock(network, weightMap, *bottleneck_csp36->getOutput(0), get_width(512, gw), 3, 2, 1, "model.38");
    ITensor *inputTensors39[] = {conv38->getOutput(0), sppcsp13->getOutput(0)};
    auto cat39 = network->addConcatenation(inputTensors39, 2);
    auto bottleneck_csp40 = bottleneckCSP2(network, weightMap, *cat39->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.40");
    auto conv41 = convBlock(network, weightMap, *bottleneck_csp40->getOutput(0), get_width(1024, gw), 3, 1, 1, "model.41");

    IConvolutionLayer *det0 = network->addConvolutionNd(*conv29->getOutput(0), 4 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.42.m.0.weight"], weightMap["model.42.m.0.bias"]);
    IConvolutionLayer *det1 = network->addConvolutionNd(*conv33->getOutput(0), 4 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.42.m.1.weight"], weightMap["model.42.m.1.bias"]);
    IConvolutionLayer *det2 = network->addConvolutionNd(*conv37->getOutput(0), 4 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.42.m.2.weight"], weightMap["model.42.m.2.bias"]);
    IConvolutionLayer *det3 = network->addConvolutionNd(*conv41->getOutput(0), 4 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.42.m.3.weight"], weightMap["model.42.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2, det3);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, float& gd, float& gw)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

int main(int argc, char **argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[1]) == "-s")
    {   
        float gd = atof(argv[2]);
        float gw = atof(argv[3]);
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, gd, gw);
        assert(modelStream != nullptr);
        std::ofstream p("yolov4-p6.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d")
    {
        std::ifstream file("yolov4-p6.engine", std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov4-p6 -s depth_multiple width_multiple // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4-p6 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0)
    {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++)
    {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size())
            continue;
        for (int b = 0; b < fcount; b++)
        {
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty())
                continue;
            cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row)
            {
                uchar *uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col)
                {
                    data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++)
        {
            auto &res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++)
        {
            auto &res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++)
            {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
