import numpy as np
import triton_python_backend_utils as pb_utils
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.baseline_mean = 0.5  # Baseline mean pixel intensity (0 to 1)
        self.threshold = 0.1      # Allowable deviation
        
        # Set up logging to file
        logging.basicConfig(
            filename="drift_metrics.log",
            level=logging.INFO,
            format="%(asctime)s - %(message)s"
        )
        self.logger = logging.getLogger("drift_preprocessor")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input image
            images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()
            
            # Compute mean pixel intensity (normalize to [0, 1])
            current_mean = np.mean(images / 255.0)
            
            # Check for drift
            drift_detected = abs(current_mean - self.baseline_mean) > self.threshold
            
            # Log metrics
            self.logger.info(f"Input mean: {current_mean:.4f}, Drift detected: {drift_detected}")
            
            # Pass input unchanged
            output_tensor = pb_utils.Tensor("preprocessed_images", images)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses