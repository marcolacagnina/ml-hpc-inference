import onnx

from ml_src.export_onnx import export_model_to_onnx

def test_onnx_export_execution(tmp_path):
    """
    Smoke test for ONNX export.

    Verifies that:
    1) export function runs without crashing
    2) ONNX file is created
    3) ONNX file is valid
    """

    # temporary path: <tmp>/model.onnx
    output_file = tmp_path / "test_model.onnx"

    # run export, using input_size as SimpleMLP class
    export_model_to_onnx(str(output_file), input_size=10)

    # check file exists
    assert output_file.exists(), "ONNX file was not created"

    # check ONNX model validity
    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)
