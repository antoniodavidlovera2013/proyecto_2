from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from skimage import filters

carpeta = Path("resultados")
carpeta.mkdir(exist_ok=True)

redes = {"instagram": (1080, 1350), "twitter": (1600, 900),
         "facebook": (1200, 630), "youtube": (1280, 720)}

filtros = {"blur": ImageFilter.BLUR, "contour": ImageFilter.CONTOUR,
           "detail": ImageFilter.DETAIL, "edge enhance": ImageFilter.EDGE_ENHANCE,
           "edge enhance more": ImageFilter.EDGE_ENHANCE_MORE,
           "emboss": ImageFilter.EMBOSS, "find edges": ImageFilter.FIND_EDGES,
           "sharpen": ImageFilter.SHARPEN, "smooth": ImageFilter.SMOOTH}


class ImagenNoCargadaException(Exception):
    def __init__(self, mensaje):
        self.__mensaje = mensaje


def verificar(imagen):
    if imagen is None:
        raise ImagenNoCargadaException("Debe cargar y redimensionar una imagen primero (Opción 1).")


def abrir_imagen(ruta):
    ruta = ruta.strip()
    if "drive.google.com/file/d/" in ruta:
        ruta = "https://drive.google.com/uc?export=download&id=" + ruta.split("/d/")[1].split("/")[0]
    if ruta.startswith("http"):
        pedido = Request(ruta, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(pedido, timeout=20) as respuesta:
            return Image.open(respuesta).convert("RGB")
    return Image.open(ruta).convert("RGB")


def guardar_figura(imagenes, titulos, nombre):
    fig, ejes = plt.subplots(1, len(imagenes), figsize=(6 * len(imagenes), 6))
    for eje, imagen, titulo in zip(ejes, imagenes, titulos):
        eje.imshow(imagen, cmap="gray" if imagen.mode == "L" else None)
        eje.set_title(titulo)
        eje.axis("off")
    plt.tight_layout()
    fig.savefig(carpeta / f"{nombre}.png", bbox_inches="tight")
    plt.show()


def redimensionar(ruta, red):
    red = red.lower().strip()
    if red in redes:
        tamanio = redes[red]
    else:
        raise ValueError("Red social no válida. Opciones: Instagram, Twitter, Facebook, Youtube")
    imagen = ImageOps.contain(abrir_imagen(ruta), tamanio)
    lienzo = Image.new("RGB", tamanio, "white")
    lienzo.paste(imagen, ((tamanio[0] - imagen.width) // 2, (tamanio[1] - imagen.height) // 2))
    lienzo.save(carpeta / f"redimensionada_{red}.jpg")
    return lienzo


def contraste(imagen):
    verificar(imagen)
    ycrcb = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    ecualizada = Image.fromarray(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))

    gris_original = np.array(imagen.convert("L"))
    gris_ecualizada = np.array(ecualizada.convert("L"))

    fig, ejes = plt.subplots(2, 2, figsize=(12, 10))

    ejes[0, 0].imshow(imagen)
    ejes[0, 0].set_title("Original")
    ejes[0, 0].axis("off")

    ejes[0, 1].imshow(ecualizada)
    ejes[0, 1].set_title("Ecualizada")
    ejes[0, 1].axis("off")

    ejes[1, 0].hist(gris_original.flatten(), bins=256, range=(0, 256), color="gray")
    ejes[1, 0].set_title("Histograma original")

    ejes[1, 1].hist(gris_ecualizada.flatten(), bins=256, range=(0, 256), color="gray")
    ejes[1, 1].set_title("Histograma ecualizada")

    plt.tight_layout()
    fig.savefig(carpeta / "contraste.png", bbox_inches="tight")
    plt.show()

def filtro(imagen, nombre):
    verificar(imagen)
    filtrada = imagen.filter(filtros[nombre])
    guardar_figura([imagen, filtrada], ["Original", nombre.title()], "filtro_" + nombre.replace(" ", "_"))


def bordes(imagen):
    verificar(imagen)
    gris = cv2.GaussianBlur(np.array(imagen.convert("L")), (5, 5), 1.5)
    canny = Image.fromarray(cv2.bitwise_not(cv2.Canny(gris, 50, 150)))
    sobel = filters.sobel(gris / 255.0)
    sobel = Image.fromarray(((1 - sobel / sobel.max()) * 255).astype("uint8"))
    guardar_figura([imagen, canny, sobel], ["Original", "Canny (OpenCV)", "Sobel (skimage)"], "bordes")


def cargar():
    ruta = input("Ruta, URL o enlace de Drive de la imagen: ")
    red = input("Red social (Instagram, Twitter, Facebook, Youtube): ")
    imagen = redimensionar(ruta, red)
    plt.imshow(imagen)
    plt.axis("off")
    plt.show()
    return imagen


def menu():
    imagen = None
    seguir = True
    print("""===  BIENVENIDOS A FOTOAPP  ===""")
    while seguir:
        print("\n1. Cargar y redimensionar imagen")
        print("2. Ajustar contraste e histograma")
        print("3. Aplicar filtro")
        print("4. Resaltar bordes")
        print("5. Salir")
        opcion = input("Seleccione una opción: ").strip()
        try:
            if opcion == "1":
                try:
                    imagen = cargar()
                except Exception as error:
                    print("Error al cargar la imagen:", error)
            elif opcion == "2":
                contraste(imagen)
            elif opcion == "3":
                verificar(imagen)
                nombres = list(filtros)
                for i, nombre in enumerate(nombres, 1):
                    print(f"{i}. {nombre.title()}")
                numero = input("Número de filtro: ").strip()
                if numero.isdigit() and 1 <= int(numero) <= 9:
                    filtro(imagen, nombres[int(numero) - 1])
                else:
                    raise ValueError("Elija un número del 1 al 9.")
            elif opcion == "4":
                bordes(imagen)
            elif opcion == "5":
                print("Gracias por utilizar FOTOAPP.")
                seguir = False
            else:
                print("Opción incorrecta.")
        except ImagenNoCargadaException as error:
            print(error)
            if input("¿Cargar una imagen ahora? (s/n): ").strip().lower() == "s":
                imagen = cargar()
        except Exception as error:
            print("Error:", error)

if __name__ == "__main__":
    menu()