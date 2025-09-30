use image::{DynamicImage, ImageBuffer, Rgb, ImageReader};
use imageproc::map::map_pixels;
use std::env;
use tract_onnx::prelude::*;

struct ImageProcessor {
    image: DynamicImage,
}

impl ImageProcessor {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("讀取圖片: {}", path);
        let img = ImageReader::open(path)?.decode()?;
        println!("原始圖片尺寸: {}x{}", img.width(), img.height());
        Ok(ImageProcessor { image: img })
    }

    pub fn resize(&mut self, scale: u32) -> &mut Self {
        let new_width = self.image.width() * scale;
        let new_height = self.image.height() * scale;
        self.image = self.image.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
        println!("放大後圖片尺寸: {}x{}", self.image.width(), self.image.height());
        self
    }

    pub fn sharpen(&mut self, intensity: f32) -> &mut Self {
        println!("使用傳統清晰化強度: {}", intensity);
        let kernel = create_sharpen_kernel(intensity);
        let rgb_image = self.image.to_rgb8();
        let sharpened_buffer: ImageBuffer<Rgb<u8>, _> = map_pixels(&rgb_image, |x, y, pixel| {
            if x == 0 || x == rgb_image.width() - 1 || y == 0 || y == rgb_image.height() - 1 {
                return pixel;
            }
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            for i in 0..3u32 {
                for j in 0..3u32 {
                    let neighbor = rgb_image.get_pixel(x + j - 1, y + i - 1);
                    let weight = kernel[i as usize][j as usize];
                    r += neighbor[0] as f32 * weight;
                    g += neighbor[1] as f32 * weight;
                    b += neighbor[2] as f32 * weight;
                }
            }
            Rgb([r.clamp(0.0, 255.0) as u8, g.clamp(0.0, 255.0) as u8, b.clamp(0.0, 255.0) as u8])
        });
        self.image = DynamicImage::ImageRgb8(sharpened_buffer);
        self
    }

    pub fn sharpen_ai(&mut self, model_path: &str) -> Result<&mut Self, Box<dyn std::error::Error>> {
        println!("使用 AI 模型進行增強 (CPU 模式，使用 tract-onnx)... ");
        println!("模型路徑: {}", model_path);

        let original_width = self.image.width();
        let original_height = self.image.height();
        let final_width = original_width * 2;
        let final_height = original_height * 2;

        // --- 1. 圖片預處理 ---
        // 為了符合模型要求，我們需要一個固定的輸入尺寸，這裡我們用 128x128 作為範例
        // 注意：不同的模型可能需要不同的固定尺寸
        let model_input_size = 128;
        let image_for_model = self.image.resize_exact(model_input_size, model_input_size, image::imageops::FilterType::Lanczos3);

        let rgb_image = image_for_model.to_rgb8();
        let input_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, model_input_size as usize, model_input_size as usize), |(_, c, y, x)| {
            rgb_image.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
        }).into();

        // --- 2. 載入 ONNX 模型並執行推論 (CPU) ---
        println!("正在載入 ONNX 模型...");
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;

        println!("執行模型推論...");
        let result = model.run(tvec!(input_tensor.into()))?;

        // --- 3. 結果後處理 ---
        let output_tensor = result[0].to_array_view::<f32>()?;
        let ai_output_shape = output_tensor.shape();
        let ai_height = ai_output_shape[2];
        let ai_width = ai_output_shape[3];

        let mut ai_output_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(ai_width as u32, ai_height as u32);
        for y in 0..ai_height {
            for x in 0..ai_width {
                let r = (output_tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (output_tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (output_tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                ai_output_image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        println!("AI 模型輸出尺寸: {}x{}", ai_width, ai_height);

        // --- 4. 將 AI 輸出的圖片縮放至目標的 2x 尺寸 ---
        let final_image = image::DynamicImage::ImageRgb8(ai_output_image)
            .resize_exact(final_width, final_height, image::imageops::FilterType::Lanczos3);

        println!("最終輸出尺寸 (2x): {}x{}", final_width, final_height);
        self.image = final_image;
        Ok(self)
    }

    pub fn save(&self, output_path: &str) -> image::ImageResult<()> {
        self.image.save(output_path)
    }
}

fn create_sharpen_kernel(intensity: f32) -> [[f32; 3]; 3] {
    if intensity <= 0.0 {
        return [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    }
    let neighbor_weight = -intensity;
    let center_weight = 1.0 - (8.0 * neighbor_weight);
    [[neighbor_weight, neighbor_weight, neighbor_weight], [neighbor_weight, center_weight, neighbor_weight], [neighbor_weight, neighbor_weight, neighbor_weight]]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "--onnx" {
        if args.len() != 5 {
            eprintln!("AI 模式參數數量錯誤！");
            eprintln!("使用方式: cargo run -- --onnx <模型路徑> <輸入圖片> <輸出圖片>");
            return Err("Invalid arguments for --onnx mode".into());
        }
        let model_path = &args[2];
        let input_path = &args[3];
        let output_path = &args[4];
        ImageProcessor::new(input_path)?
            .sharpen_ai(model_path)?
            .save(output_path)?;
    } else {
        if args.len() != 4 {
            eprintln!("傳統模式參數數量錯誤！");
            eprintln!("使用方式: cargo run -- <輸入圖片> <輸出圖片> <清晰化強度>");
            eprintln!("範例:     cargo run -- test.png renewed.png 1.5");
            return Err("Invalid arguments".into());
        }
        let input_path = &args[1];
        let output_path = &args[2];
        let sharpen_intensity: f32 = match args[3].parse() {
            Ok(num) => num,
            Err(_) => {
                eprintln!("錯誤: 清晰化強度 '{}' 必須是一個數字。", args[3]);
                return Err("Invalid intensity value".into());
            }
        };
        ImageProcessor::new(input_path)?
            .resize(2)
            .sharpen(sharpen_intensity)
            .save(output_path)?;
    }
    println!("圖片處理完成！已成功儲存。");
    Ok(())
}