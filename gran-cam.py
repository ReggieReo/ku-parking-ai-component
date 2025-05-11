from YOLOv8_Explainer import yolov8_heatmap, display_images

model = yolov8_heatmap(
    weight="/Users/reoreggie/Documents/project/mlartifacts/345266831538558357/71fa5455ea8e4bfda93abb0b28b24ad3/artifacts/weights/best.pt", 
        conf_threshold=0.4,  
        method = "EigenGradCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        ratio=0.02,
        # show_box=True,
        show_box=False,
        renormalize=False,
)

imagelist = model(
    img_path="dataset/images/train/SCR-20250509-ttlk_jpeg.rf.ccadbb9b922d3563d657e1d1fa6f13e7.jpg", 
    )

display_images(imagelist)
