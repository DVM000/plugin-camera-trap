# Animal identification from camera-trap images

This app simulares the integration of a classification network for animal identification in camera traps. Camera traps usually
take a burst of images when passive infrared sensor activates.
 
The Convolutional Neural Network is light-weight since it was trained for facilitating its integration on embedded devices. It
provides two categories: *animal* vs. *blank*. Trained on [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti) dataset,
 this simple network achieved 0.80 accuracy on validation dataset. However, when deployed at different locations, accuracy can
 highly drop, being necessary on-device training with specific-location data.
 
 
## How to Use
To run the program,

```bash
# Captures and publishes detections and images containing animals 
python3 main.py --stream bottom_camera --n 10
```

this will capture 10 images and publish animal detections on topic `env.detection.animal`. Value for a topic indicates the confidence of the detection.
Images containing animals are also uploaded. 

