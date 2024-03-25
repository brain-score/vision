class Stimulus:
    def from_path(self, path):
        raise NotImplementedError("Choose a concrete Stimulus type to use.")
    
    @staticmethod
    def is_video_path(path):
        extension = path.split('.')[-1]
        return extension in ['mp4', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mkv', 'gif']
    
    @staticmethod
    def is_image_path(path):
        extension = path.split('.')[-1]
        return extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        