from abc import ABC, abstractmethod

class AbstractVideoProcessor(ABC):

    @abstractmethod
    def process(self, frame, fps=1e-6):
        """
        Abstract method for video processing
        
        Args:
            frame (array): The current frame for detection.
            fps (float): Current FPS
        
        Returns:
            Processed frame
        """
        pass
        