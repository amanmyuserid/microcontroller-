#include "esp_camera.h"
#include <string.h>
#include <math.h>


#include "SD.h"
#include "FS.h"
#include "SD_MMC.h" 



#include <TensorFlowLite_ESP32.h>

#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/c_api_types.h"


#define wid  60*5
#define hei  45                                      
#define cha  1

//int8_t img_a[60*80];
//int8_t img_b[60*80];
//int8_t img_c[60*80];
//int8_t img_d[60*80];
//int8_t img_e[60*80];
//int8_t img_f[60*80];


int8_t final_inp_arr[1*hei*wid*cha]; 
fs::FS sdcard = SD_MMC;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;//
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;//
TfLiteTensor* input = nullptr;
TfLiteTensor* output_tensor = nullptr;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 90 * 1024;
static uint8_t tensor_arena[kTensorArenaSize]; 
}


int frame_num = 1;
float input_scale;
float input_zero_point;
float norm_val;

float output_scale;
float output_zero_point;


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
config.pixel_format = PIXFORMAT_GRAYSCALE;
config.frame_size = FRAMESIZE_QQVGA; // QQVGA = 160*120
config.jpeg_quality = 10;
config.fb_count = 1;



esp_err_t err = esp_camera_init(&config);
if (err != ESP_OK) {
  // Handle error
}


// Initialize the SD card
  if (!SD_MMC.begin()) {
    Serial.println("SD card initialization failed!");
    return;
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


  static tflite::MicroMutableOpResolver<19> micro_mutable_op_resolver;
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
  micro_mutable_op_resolver.AddLogistic();

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

  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;

  norm_val = 1./255; // 1/255 if you rescaling at input time else put 1.0 only 


  output_tensor = interpreter->output(0);
  output_scale = output_tensor->params.scale;
  output_zero_point = output_tensor->params.zero_point;
    
}




void loop() {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
  

  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    // Handle error
  }

//  printf("\n input_scale %f, input_zero_point %f, norm_val %f \n", input_scale, input_zero_point, norm_val);
//  printf("\n output_scale %f, output_zero_point %f \n", output_scale, output_zero_point);
//  int8_t final_inp_arr[1*hei*wid*cha];

//  float input_scale = input->params.scale;
//  float input_zero_point = input->params.zero_point;
//
//  float norm_val = 1./255; // 1/255 if you rescaling at input time else put 1.0 only 

  uint8_t* image_data = (uint8_t*) fb->buf;
  // final_inp_arr 
  if (frame_num <= 5){
      
      
//      int pix_num = 0;
//      int window_start = (frame_num-1)*45*60;
//      for(int j= 0; j < 45*60; j++){
//
//          uint8_t pix_val = image_data[pix_num];
//          
//          float normalized_pix_val = (float) (pix_val / 127.5) - 1.0;
//          
//          int8_t quantized_pix_val = (int8_t) round(normalized_pix_val * 127);
//                    
//
//          final_inp_arr[window_start+j] = quantized_pix_val;
//
//          pix_num = pix_num + 2;       
//        
//      }

      int pix_count = 0;
      int window_start = (frame_num-1)*45*60;
      float row_idx = 0;
      for (int i = 0; i < 45; i += 1) {
          float col_idx = 0;
          for (int j = 0; j < 60; j += 1) {
      
              // Get the grayscale value of the pixel in the original image
              uint8_t pix_val = fb->buf[((int)(row_idx))*160 + (int)col_idx];
      
//            float pix_val_float = (pix_val*norm_val / input_scale) + input_zero_point;
//
//            int8_t pix_val_int8; 
//            if(pix_val_float < -128){
//              pix_val_int8 = -128;
//            }else if(pix_val_float > 127){
//              pix_val_int8 = 127;
//            }else{
//              pix_val_int8 = pix_val_float;
//            }
            
            
            final_inp_arr[(5-1)*60 +i*300 + j] = (pix_val*norm_val / input_scale) + input_zero_point;
              col_idx = col_idx+ (float)160/60;
              pix_count++;
          }
          row_idx = row_idx + (float)120/45;
      }      
      

      frame_num++;
//      continue;
  }else{

    // start : shifting the pixel value to one windows back
    
//    for(int w = 0; w < 4; w++){
//      for(int i = 0; i<45*60; i++){
//        final_inp_arr[w*45*60+i] = final_inp_arr[(w+1)*45*60+i];
//      }
//    }  


    for(int i= 0; i < 300*45; i = i + 300){
      for (int j = 60 ; j < 300 ; j++){
        final_inp_arr[i+(j-60)] = final_inp_arr[i+j];
      }
    }

    
      
    // end : shifting the pixel value to one windows back

    // start : inserting new image in last window of final input array
    
    int pix_count = 0;
    int window_start = 4*45*60;
    float row_idx = 0;
    for (int i = 0; i < 45; i += 1) {
        float col_idx = 0;
        for (int j = 0; j < 60; j += 1) {
    
            // Get the grayscale value of the pixel in the original image
            uint8_t pix_val = fb->buf[((int)(row_idx))*160 + (int)col_idx];

//            float pix_val_float = (pix_val*norm_val / input_scale) + input_zero_point;
//
//            int8_t pix_val_int8; 
//            if(pix_val_float < -128){
//              pix_val_int8 = -128;
//            }else if(pix_val_float > 127){
//              pix_val_int8 = 127;
//            }else{
//              pix_val_int8 = pix_val_float;
//            }
            
            
            final_inp_arr[(5-1)*60 +i*300 + j] = (pix_val*norm_val / input_scale) + input_zero_point;
            col_idx = col_idx+ (float)160/60;
            pix_count++;
        }
        row_idx = row_idx + (float)120/45;
    } 
    // end : inserting new image in last window of final input array   


    // start : Inference start



    // TfLiteTensor* input = interpreter->input(0);
//    /input->data.int8 = &final_inp_arr[0];

    for(int i = 0; i<45*300; i++){
      input->data.int8[i] = final_inp_arr[i];
    }
    
     interpreter->Invoke();
    // TfLiteTensor* output = interpreter->output(0);
    
//     TfLiteTensor* output_tensor = interpreter->output(0);
    
      const int number_of_classes = output_tensor->bytes / sizeof(int8_t);
     //  const int output_count = output_tensor_size / sizeof(int8_t);
    
      printf("\n=========================================================\n");
      for (int i = 0; i < number_of_classes; i++) {
        printf("class%d %d, ", i+1, output_tensor->data.int8[i]);
      }
      printf("\n");
      
      
//      float output_scale = output_tensor->params.scale;
//      int output_zero_point = output_tensor->params.zero_point;
                                                                                                                   
      
      // Create a new float array to store the converted output values
      float output_float[number_of_classes];
    
//      printf("Output converted from int8 ablove to float below\n");
      // Convert int8 output to float
      for (int i = 0; i < number_of_classes; i++) {
          output_float[i] = ((float)output_tensor->data.int8[i] - (float)output_zero_point) * output_scale; 
          printf("class%d %f, ", i+1, output_float[i]);
      }

//      int max_val = 0, max_val_idx;
//      for(int i = 0; i<number_of_classes; i++){
//        if(max_val < output_float[i]){
//          max_val = output_float[i];
//          max_val_idx = i+1;
//        }
//        
//      }
//      if(max_val_idx == 1){
//        printf("Prediction : Pipe going inside");
//      }else{
//        printf("Prediction : Nothing "); 
//      }
      
      
//      Serial.print("\nRunning...\n");    

    // end : Inference end

    // Save the image data to a text file
    printf("\n frame no %d \n", frame_num);
    if(frame_num == 20){
      saveImageToFile("/glarus_image_45_300.txt");
    }
    frame_num++;
  }    
  

  esp_camera_fb_return(fb);


}

void saveImageToFile(String fileName) {
  // Open the file for writing
  File file = sdcard.open(fileName, FILE_WRITE);
  if (!file) {
    printf("\nFailed to create file!\n");
    return;
  }

  // Write the image data to the file
  for (int i = 0; i < 45; i++) {
    for (int j = 0; j < 300; j++) {
      file.print(final_inp_arr[i*300 + j]);
      file.print(" ");
    }
    file.println(); // move to the next row
  }

  // Close the file
  file.close();

  printf("Image saved to file");
}

