#include <stdio.h>
#include <inttypes.h>
#include "esp_spiffs.h"
#include "sdkconfig.h"
#include "esp_err.h"
#include "esp_log.h"
#include <time.h>
#include <driver/i2c.h>
#include <string>
extern "C"
{
#include "llama.h"
#include "llm.h"
#include <u8g2.h>
#include "u8g2_esp32_hal.h"
}
#define XPOWERS_CHIP_AXP2101
#include "XPowersLib.h"

static const char *TAG = "MAIN";
u8g2_t u8g2;
static XPowersPMU power;

#define PIN_SDA (gpio_num_t)7
#define PIN_SCL (gpio_num_t)6
#define OLED_I2C_ADDRESS 0x78



/**
 * @brief intializes SPIFFS storage
 *
 */
void init_storage(void)
{

    ESP_LOGI(TAG, "Initializing SPIFFS");

    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/data",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = false};

    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        }
        else if (ret == ESP_ERR_NOT_FOUND)
        {
            ESP_LOGE(TAG, "Failed to find SPIFFS partition");
        }
        else
        {
            ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
        }
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to get SPIFFS partition information (%s)", esp_err_to_name(ret));
    }
    else
    {
        ESP_LOGI(TAG, "Partition size: total: %d, used: %d", total, used);
    }
}



/**
 * @brief Callbacks once generation is done
 *
 * @param tk_s The number of tokens per second generated
 */
void generate_complete_cb2(float tk_s)
{
    char buffer[50];
    sprintf(buffer, "%.2f tok/s", tk_s);

}

/**
 * @brief Callbacks for token flow
 *
 * @param tokens The tokens generated
 */
void output_cb2(char* token)
{
    // buffer one row
    // use u8g2 print
    static std::string buffer;
    static int buf_pos = 0;
    static int row = 0;
    buffer += token;
    buf_pos += strlen(token);

    printf("%s", buffer.c_str());
    buffer.clear();

}

/*void set_prompt(char *out){
    char buffer2[100];
    fscanf(stdin, "%s", buffer2);
    strcpy(out, buffer2);
}*/

extern "C" void app_main()
{
    //init_display();
    //write_display("Loading Model");
    init_storage();

    // default parameters
    char *checkpoint_path = "/data/stories260K.bin"; // e.g. out/model.bin
    char *tokenizer_path = "/data/tok512.bin";
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;                 // number of steps to run for
    char *prompt = NULL;             // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    Transformer transformer;
    ESP_LOGI(TAG, "LLM Path is %s", checkpoint_path);
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    char *buffer_prompt = "Once upon a time, there was an angel that";
    // run!
    //draw_llama();
    //u8g2_ClearBuffer(&u8g2);
    generate(&transformer, &tokenizer, &sampler, buffer_prompt, steps, &generate_complete_cb2, &output_cb2);

    free(buffer_prompt);
}
