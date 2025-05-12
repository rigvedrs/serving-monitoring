import numpy as np
import triton_python_backend_utils as pb_utils
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.baseline_mean = 0.8  # Baseline mean confidence score
        self.threshold = 0.1      # Allowable deviation
        
        # Set up logging to file
        logging.basicConfig(
            filename="drift_metrics.log",
            level=logging.INFO,
            format="%(asctime)s - %(message)s"
        )
        self.logger = logging.getLogger("drift_postprocessor")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get YOLO output
            output0 = pb_utils.get_input_tensor_by_name(request, "output0").as_numpy()
            
            # Extract confidence scores (assuming row 4 contains confidences)
            confidences = output0[4, :]
            confidences = confidences[confidences > 0]  # Filter valid detections
            
            # Compute mean confidence
            current_mean = np.mean(confidences) if confidences.size > 0 else 0.0
            
            # Check for drift
            drift_detected = abs(current_mean - self.baseline_mean) > self.threshold
            
            # Log metrics
            self.logger.info(f"Output mean confidence: {current_mean:.4f}, Drift detected: {drift_detected}")
            
            # Pass output unchanged
            output_tensor = pb_utils.Tensor("output0", output0)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses