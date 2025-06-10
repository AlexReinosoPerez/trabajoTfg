import os
import time
import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def save_image(image_url, save_path):
    """Descarga y guarda una imagen desde una URL."""
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            img.save(save_path, quality=95)  # Guardar con alta calidad
            print(f"Imagen guardada: {save_path}")
    except Exception as e:
        print(f"Error al guardar la imagen {save_path}: {e}")

def scroll_slowly(driver, steps=25, delay=0.5):
    """Hace un scroll progresivo más corto pero más veces para cargar todas las imágenes."""
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    step = scroll_height // steps  # Más pasos, desplazamientos más cortos
    current_position = 0

    for _ in range(steps):
        current_position += step
        driver.execute_script(f"window.scrollTo(0, {current_position});")
        time.sleep(delay)  # Dar tiempo para que carguen las imágenes

def get_images_from_url(driver, url, folder_name, max_images=2000):
    """Obtiene todas las imágenes desde la página web proporcionada y navega entre páginas."""
    try:
        driver.get(url)
        time.sleep(5)  # Esperar carga inicial

        # Crear una carpeta en el escritorio para guardar las imágenes
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        save_folder = os.path.join(desktop_path, "imagenes_descargadas", folder_name)
        os.makedirs(save_folder, exist_ok=True)

        # Contador de imágenes descargadas y conjunto para evitar duplicados
        image_count = 0
        downloaded_images = set()
        page_number = 1
        no_new_images_count = 0  # Control para evitar bucles infinitos

        while image_count < max_images:
            print(f"Descargando imágenes de la página {page_number}...")

            # Hacer scroll progresivo con más pasos
            scroll_slowly(driver, steps=25, delay=0.5)

            # Capturar todas las imágenes después de cargar completamente la página
            images = driver.find_elements(By.CSS_SELECTOR, "img.ng-isolate-scope")

            for img in images:
                try:
                    img_url = img.get_attribute('src')
                    if img_url and img_url.endswith(".jpg") and img_url not in downloaded_images:
                        image_count += 1
                        downloaded_images.add(img_url)
                        save_path = os.path.join(save_folder, f"imagen_{image_count}.jpg")
                        save_image(img_url, save_path)

                        if image_count >= max_images:
                            print(f"Se alcanzó el máximo de {max_images} imágenes.")
                            return  # Salir de la función cuando se alcance el límite
                except Exception as e:
                    print(f"Error al procesar una imagen: {e}")

            # Intentar cargar más imágenes con el botón "LOAD MORE"
            try:
                load_more_button = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "masonry-load-more-button"))
                )
                load_more_button.click()
                time.sleep(5)  # Esperar la carga de nuevas imágenes
                page_number += 1
                continue  # Repetir el ciclo después de cargar más imágenes
            except:
                pass  # Si el botón no existe, intentar pasar de página normalmente

            # Si no hay botón de "LOAD MORE", intentar ir a la siguiente página
            next_buttons = driver.find_elements(By.CSS_SELECTOR, "a[rel='next']")
            if next_buttons:
                next_buttons[0].click()
                time.sleep(5)
                page_number += 1
                continue  # Repetir el ciclo después de pasar de página
            
            # Si no se encontraron más imágenes ni forma de avanzar, detenerse
            no_new_images_count += 1
            if no_new_images_count >= 3:
                print("No se encontraron nuevas imágenes después de 3 intentos. Deteniendo...")
                break

        print(f"Se descargaron un total de {image_count} imágenes en {folder_name}.")
    except Exception as e:
        print(f"Error al procesar la URL {url}: {e}")

# URLs de las páginas que deseas scrapear
urls = {
    "Impresionismo": "https://www.wikiart.org/en/paintings-by-style/impressionism?select=featured",
    "Renacimiento": "https://www.wikiart.org/en/paintings-by-style/northern-renaissance?select=featured",
    "Pop Art": "https://www.wikiart.org/en/paintings-by-style/pop-art?select=featured"
}

# Configurar el controlador de Chrome
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Iterar sobre los enlaces de las páginas y obtener imágenes
for name, url in urls.items():
    print(f"Descargando imágenes de {name}...")
    get_images_from_url(driver, url, name, max_images=2000)
    print(f"Proceso completado para {name}.")

# Cerrar el navegador al finalizar
driver.quit()
