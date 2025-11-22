import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def fetch_image_cv2(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)       
        if img_bgr is None:
            print("Nie udało się zdekodować obrazu.")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Błąd pobierania: {e}")
        return None

def plot_histograms_cv2(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[0].set_title("Oryginalne zdjęcie")
    ax[0].axis('off')
    hist_lum = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ax[1].plot(hist_lum, color='black')
    ax[1].fill_between(range(256), hist_lum.flatten(), color='gray', alpha=0.5)
    ax[1].set_title("Histogram Jasności")
    ax[1].set_xlim([0, 256])
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax[2].plot(hist, color=color, label=f'Kanał {color.upper()}')
    ax[2].set_title("Histogramy RGB")
    ax[2].set_xlim([0, 256])
    ax[2].legend()
    ax[3].axis('off')
    ax[3].text(0.05, 0.5, "Analiza jakości (patrz konsola)", fontsize=12)

    plt.tight_layout()
    plt.show()

def analyze_quality_cv2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_val, std_dev = cv2.meanStdDev(gray)
    mean_val = mean_val[0][0]
    std_dev = std_dev[0][0]
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    pixels_black = hist[0][0]   # Piksele o wartości 0
    pixels_white = hist[255][0] # Piksele o wartości 255
    total_pixels = image.shape[0] * image.shape[1]
    clipping_shadows = (pixels_black / total_pixels) * 100
    clipping_highlights = (pixels_white / total_pixels) * 100
    diagnosis = []
    needs_fix = False

    if mean_val < 60:
        diagnosis.append("ZDJĘCIE NIEDOŚWIETLONE (Za ciemne).")
        needs_fix = True
    elif mean_val > 195:
        diagnosis.append("ZDJĘCIE PRZEŚWIETLONE (Za jasne).")
        needs_fix = True
    else:
        diagnosis.append("Ekspozycja poprawna.")

    if std_dev < 30:
        diagnosis.append("BARDZO NISKI KONTRAST (Zdjęcie 'mgliste').")
        needs_fix = True
    elif std_dev < 50:
        diagnosis.append("Umiarkowany kontrast.")
    else:
        diagnosis.append("Dobry kontrast.")

    if clipping_shadows > 1.0:
        diagnosis.append(f" Utrata detali w cieniach ({clipping_shadows:.1f}% czerni absolutnej).")
    if clipping_highlights > 1.0:
        diagnosis.append(f" Utrata detali w światłach ({clipping_highlights:.1f}% bieli absolutnej).")

    return {
        "mean": mean_val,
        "std": std_dev,
        "diagnosis": diagnosis,
        "needs_fix": needs_fix
    }

def improve_image_cv2_clahe(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_improved = clahe.apply(l)
    lab_improved = cv2.merge((l_improved, a, b))
    result = cv2.cvtColor(lab_improved, cv2.COLOR_LAB2RGB)
    return result

url = "https://upload.wikimedia.org/wikipedia/commons/c/c0/Foggy_morning_at_Twin_Peaks.jpg" 

print(f"Pobieranie: {url} ...")
img = fetch_image_cv2(url)

if img is not None:
    plot_histograms_cv2(img)
    print("\n---WYNIKI ANALIZY---")
    stats = analyze_quality_cv2(img)
    print(f"Średnia jasność: {stats['mean']:.2f}")
    print(f"Kontrast (StdDev): {stats['std']:.2f}")
    print("Diagnoza:")
    for d in stats['diagnosis']:
        print(f"  {d}")
    if stats['needs_fix'] or True: 
        print("\n---WYKRYTO PROBLEMY - URUCHAMIANIE KOREKCJI---")
        fixed_img = improve_image_cv2_clahe(img)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title("PRZED (Oryginał)")
        axes[0].axis('off')
        axes[1].imshow(fixed_img)
        axes[1].set_title("PO (OpenCV CLAHE)")
        axes[1].axis('off')
        plt.show()