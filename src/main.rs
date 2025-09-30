use image::{ImageReader, Rgb, ImageBuffer};
use imageproc::map::map_pixels;
use std::env;

/// 根據指定的強度，動態產生一個用於清晰化的 3x3 卷積核心。
///
/// # Arguments
/// * `intensity` - 清晰化強度。0.0 代表不處理，1.0 為標準強度，愈大愈強。
///
/// # Returns
/// 一個 3x3 的浮點數陣列，其權重總和為 1。
fn create_sharpen_kernel(intensity: f32) -> [[f32; 3]; 3] {
    if intensity <= 0.0 {
        // 強度為 0 或負數時，返回一個「什麼都不做」的核心
        return [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
    }

    // 根據強度計算周圍像素的負權重
    let neighbor_weight = -intensity;
    // 根據「總和為 1」的原則，計算中心權重
    let center_weight = 1.0 - (8.0 * neighbor_weight);

    [
        [neighbor_weight, neighbor_weight, neighbor_weight],
        [neighbor_weight, center_weight,   neighbor_weight],
        [neighbor_weight, neighbor_weight, neighbor_weight],
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 讀取並解析命令列參數
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        // eprintln! 會將訊息印到標準錯誤輸出，適合用於錯誤提示
        eprintln!("參數數量錯誤！");
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

    // 2. 讀取圖片
    println!("讀取圖片: {}", input_path);
    let img = ImageReader::open(input_path)?.decode()?;
    println!("原始圖片尺寸: {}x{}", img.width(), img.height());

    // 3. 圖片放大 (放大兩倍)
    let new_width = img.width() * 2;
    let new_height = img.height() * 2;
    let resized_img = img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
    println!("放大後圖片尺寸: {}x{}", resized_img.width(), resized_img.height());

    // 4. 圖片清晰化
    println!("使用清晰化強度: {}", sharpen_intensity);
    let kernel = create_sharpen_kernel(sharpen_intensity);
    let rgb_image = resized_img.to_rgb8();
    
    let sharpened_img: ImageBuffer<Rgb<u8>, _> = map_pixels(&rgb_image, |x, y, pixel| {
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

        Rgb([
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        ])
    });

    // 5. 儲存結果
    sharpened_img.save(output_path)?;
    println!("圖片處理完成！已儲存為 {}", output_path);

    Ok(())
}
