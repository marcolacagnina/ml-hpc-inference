import torch

def export_model_to_onnx(model, output_path: str, input_size: int, device="cpu"):
    """
    Exports a trained PyTorch model to ONNX format.

    Args:
        model: trained PyTorch model
        output_path (str): destination path for .onnx file
        input_size (int): input feature size
        device (str): device for dummy input
    """
    model.eval()

    dummy_input = torch.randn(1, input_size).to(device)

    print(f"Exporting model to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("Model exported successfully.")

