#!/bin/sh

cargo run -- --onnx ./model/Real-ESRGAN-x4plus_float/model.onnx test.png renewed_from_cli.png
# cargo run -- --onnx ./model/super_resolution.onnx test.png renewed_from_cli.png
#cargo run -- --onnx super_resolution.onnx test.png renewed_from_cli.png
