import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Title and Description
st.set_page_config(page_title="Pavement AI", layout="centered")
st.title("🚧 Pavement Condition Monitoring System")
st.write("Upload a road image to detect cracks, potholes, and distress.")

# 2. Sidebar for Model Settings
st.sidebar.header("Model Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# 3. Load the Model (Cache it so it doesn't reload every time)
@st.cache_resource
def load_model():
    # REPLACE this path with your actual 'best.pt' path
    # If running locally, put best.pt in the same folder as app.py
    return YOLO("best.pt")

try:
    model = load_model()
    st.sidebar.success("Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 4. Image Upload Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 5. Run Inference when button is clicked
    if st.button("Analyze Pavement"):
        with st.spinner("Analyzing..."):
            # Convert PIL image to numpy for YOLO
            img_array = np.array(image)
            
            # Run the model
            results = model.predict(img_array, conf=conf_threshold)
            
            # Plot results on the image
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted)
            
            # Display Result
            st.image(res_image, caption='Detected Defects', use_container_width=True)
            
            # 6. Show Statistics (The "Engineering" Part)
            # Count the number of boxes per class
            boxes = results[0].boxes
            if len(boxes) > 0:
                cls_ids = boxes.cls.cpu().numpy()
                names = model.names
                
                st.subheader("⚠️ Distress Report")
                distress_counts = {}
                for cls_id in cls_ids:
                    name = names[int(cls_id)]
                    distress_counts[name] = distress_counts.get(name, 0) + 1
                
                # specific metrics for resume impact
                col1, col2 = st.columns(2)
                for name, count in distress_counts.items():
                    col1.metric(label=name, value=f"{count} Detected")
                
                # Simple Logic for "Road Health Score"
                total_defects = len(boxes)
                if total_defects == 0:
                    health_status = "Excellent"
                    color = "green"
                elif total_defects < 3:
                    health_status = "Fair (Maintenance Recommended)"
                    color = "orange"
                else:
                    health_status = "Critical (Immediate Repair Needed)"
                    color = "red"
                    
                st.markdown(f"### Overall Rating: :{color}[{health_status}]")
                
            else:

                st.success("✅ No defects detected. Road is in good condition.")
