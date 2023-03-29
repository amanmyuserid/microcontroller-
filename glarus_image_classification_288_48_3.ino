#include "esp_camera.h"
#include <string.h>
#include <math.h>


#include <TensorFlowLite_ESP32.h>

#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/c_api_types.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;//
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;//
TfLiteTensor* input = nullptr;


// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 90 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}


void setup() {
  
Serial.begin(115200);//
camera_config_t config;
config.ledc_channel = LEDC_CHANNEL_0;
config.ledc_timer = LEDC_TIMER_0;
config.pin_d0 = 5;//CAM_PIN_D0;
config.pin_d1 = 18;//CAM_PIN_D1;
config.pin_d2 = 19;//CAM_PIN_D2;
config.pin_d3 = 21;//CAM_PIN_D3;
config.pin_d4 = 36;//CAM_PIN_D4;
config.pin_d5 = 39;//CAM_PIN_D5;
config.pin_d6 = 34;//CAM_PIN_D6;
config.pin_d7 = 35;//CAM_PIN_D7;
config.pin_xclk = 0;//CAM_PIN_XCLK;
config.pin_pclk = 22;//CAM_PIN_PCLK;
config.pin_vsync = 25;//CAM_PIN_VSYNC;
config.pin_href = 23;//CAM_PIN_HREF;
config.pin_sscb_sda = 26;//CAM_PIN_SIOD;
config.pin_sscb_scl = 27;//CAM_PIN_SIOC;
config.pin_pwdn = 32;//CAM_PIN_PWDN;
config.pin_reset = -1;//CAM_PIN_RESET;
config.xclk_freq_hz = 20000000;//XCLK_FREQ;//10000000;
config.pixel_format = PIXFORMAT_RGB565;
config.frame_size = FRAMESIZE_96X96;
config.jpeg_quality = 10;
config.fb_count = 1;



esp_err_t err = esp_camera_init(&config);
if (err != ESP_OK) {
  // Handle error
}



// Tensorflow
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
//    error_reporter->Report(
//        "Model provided is schema version %d not equal "
//        "to supported version %d.",
//        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }


  static tflite::MicroMutableOpResolver<18> micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddAveragePool2D();
  micro_mutable_op_resolver.AddConv2D();
  micro_mutable_op_resolver.AddDepthwiseConv2D();
  micro_mutable_op_resolver.AddReshape();
  micro_mutable_op_resolver.AddSoftmax();
  micro_mutable_op_resolver.AddShape();
  micro_mutable_op_resolver.AddAdd();
  micro_mutable_op_resolver.AddPad();
  micro_mutable_op_resolver.AddMean();
  micro_mutable_op_resolver.AddMul();
  micro_mutable_op_resolver.AddFullyConnected();
  micro_mutable_op_resolver.AddAdd();
  micro_mutable_op_resolver.AddStridedSlice();
  micro_mutable_op_resolver.AddPack();
  micro_mutable_op_resolver.AddMaxPool2D();
  micro_mutable_op_resolver.AddConcatenation();
  micro_mutable_op_resolver.AddBroadcastTo();
  micro_mutable_op_resolver.AddFill();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");//
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

    
}

void loop() {



  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    // Handle error
  }



uint16_t* image_data = (uint16_t*) fb->buf;
const int IMAGE_SIZE = 96 * 96 ;
const int INPUT_SIZE = IMAGE_SIZE * 3;
int8_t* input_data = (int8_t*) fb->buf;
for (int i = 0; i < IMAGE_SIZE ; i++) {
    // Extract R, G, and B components from 16-bit RGB565 pixel value
    uint8_t red = (image_data[i] & 0xF800) >> 11;
    uint8_t green = (image_data[i] & 0x07E0) >> 5;
    uint8_t blue = (image_data[i] & 0x001F);
    
    // Normalize R, G, and B pixel values to range [-1, 1]
    float normalized_red = (float) (red / 127.5) - 1.0;
    float normalized_green = (float) (green / 127.5) - 1.0;
    float normalized_blue = (float) (blue / 127.5) - 1.0;

    // Quantize normalized R, G, and B pixel values to int8_t
    int8_t quantized_red = (int8_t) round(normalized_red * 127);
    int8_t quantized_green = (int8_t) round(normalized_green * 127);
    int8_t quantized_blue = (int8_t) round(normalized_blue * 127);

    // Write quantized R, G, and B pixel values directly to input buffer
    input_data[3*i] = quantized_red;
    input_data[3*i+1] = quantized_green;
    input_data[3*i+2] = quantized_blue;
}





// TfLiteTensor* input = interpreter->input(0);/
 input->data.int8 = input_data;

 interpreter->Invoke();
// TfLiteTensor* output = interpreter->output(0);

    TfLiteTensor* output_tensor = interpreter->output(0);

  const int number_of_classes = output_tensor->bytes / sizeof(int8_t);
//  const int output_count = output_tensor_size / sizeof(int8_t);

  printf("\n=========================================================\n");
  for (int i = 0; i < number_of_classes; i++) {
    printf("class%d %d, ", i+1, output_tensor->data.int8[i]);
  }
  printf("\n");
  
  float output_scale = output_tensor->params.scale;
  int output_zero_point = output_tensor->params.zero_point;
  
  
  // Create a new float array to store the converted output values
  float output_float[number_of_classes];

  printf("Output converted from int8 ablove to float below\n");
  // Convert int8 output to float
  for (int i = 0; i < number_of_classes; i++) {
      output_float[i] = ((float)output_tensor->data.int8[i] - (float)output_zero_point) * output_scale; 
      printf("class%d %f, ", i+1, output_float[i]);
  }
  Serial.print("\nRunning...\n");
 esp_camera_fb_return(fb);


}
