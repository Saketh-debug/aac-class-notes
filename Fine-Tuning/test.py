from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO('best.pt')

# Run inference
results = model(r'C:\Users\Charan\OneDrive\Desktop\Projects\fine tuning example\image2.jpeg')
# results = model(r'C:\Users\Charan\OneDrive\Desktop\Projects\fine tuning example\image.png')
# Display results
for result in results:
    result.show()
    print(result.boxes)  # Print detections
