from custom_node_helper import CustomNodeHelper

MODELS = [
    "mobilenet0.25_Final.pth",
    "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
]

class ComfyUI_HeadshotPro(CustomNodeHelper):
    @staticmethod
    def models():
        return MODELS

    @staticmethod
    def add_weights(weights_to_download, node):
        if node.is_type("GetCannyFromPoseAndFace"):
            weights_to_download.extend(MODELS)

    @staticmethod
    def weights_map(base_url):
        return {
            model: {
                "url": f"{base_url}/custom_nodes/ComfyUI-HeadshotPro/{model}.tar",
                "dest": "ComfyUI/custom_nodes/ComfyUI-HeadshotPro/weights",
            }
            for model in MODELS
        }
