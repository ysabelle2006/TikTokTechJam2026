from PIL import Image

from tiktoktechjam2026.models.detector import Detector


image = Image.open("test_image.jpg")

for mode in ["srm", "fft"]:
    print()
    print("Testing detector:", mode)

    detector = Detector(
        frequency_mode=mode
    )

    probability = detector.predict(image)

    print(
        "AIGC probability:",
        probability
    )

    print(
        "Valid probability:",
        0.0 <= probability <= 1.0
    )