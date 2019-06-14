import cv2
import keras
import numpy as np
import sys

# Prvi i jedini argument komandne linije je indeks test primera
if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

tp_idx = sys.argv[1]
# Odmah učitavamo kao grayscale
img = cv2.imread('tests/{}.png'.format(tp_idx), cv2.IMREAD_GRAYSCALE)

#################################################################################
# U ovoj sekciji implementirati obradu slike, ucitati prethodno trenirani Keras
# model, i dodati bounding box-ove i imena klasa na sliku.
# Ne menjati fajl van ove sekcije.

# Ucitavamo model
model = keras.models.load_model('models/fashion_dataaug_1.h5')

# Solution konvertujemo u RGB da bi mogli da pišemo crveni tekst i crtamo plave boxove
solution = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

labels = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot"
]

# Sklonimo noise (one tackice) - parametri su preporuceni 7, 21. 50 je nabadano
dst = cv2.fastNlMeansDenoising(img, None, 50, 7, 21)

# Tresholding da izdvojimo samo predmete, inverzno jer konture rade za bele objekte na crnoj pozadini
ret, thresh = cv2.threshold(dst, 240, 255, cv2.THRESH_BINARY_INV)

# probao nesto ne radi
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(thresh, kernel,iterations = 1)

# Izvucemo konture iz thresh slike
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

final_contours = []
i = -1
for c in contours: # Prolazimo kroz sve konture
  i += 1
  area = cv2.contourArea(c)
  if hierarchy[0][i][3] != -1: # Ako je unutar druge konture ignorisi
    continue
  elif area < 20: # Ako je previse malo ignorisi
    continue
  else:
    final_contours.append(c)  

items = []
for c in final_contours:
    x, y, w, h = cv2.boundingRect(c) # Dohvatimo gornji levi ćošak predmeta kao i veličinu bounding box-a
    
    # Računamo centar mase konture da bi centrirali objekat
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    if h > w:
      x = cx - h // 2
      cv2.rectangle(solution, (x , y), (x+h, y+h), (255, 0, 0), 2) # Bounding box crtanje
      # Izdvojimo sliku koju posle šaljemo 
      item = cv2.bitwise_not(img[y : y + h, x : x + h]) # Uzimamo originalnu sliku sa sve nois-om i invertujemo pixele
    elif w > h:
      y = cy - w // 2 
      cv2.rectangle(solution, (x, y ), (x+w, y+w), (255, 0, 0), 2)
      item = cv2.bitwise_not(img[y : y + w, x : x + w])
    else:
      cv2.rectangle(solution, (x, y), (x+w, y+h), (255, 0, 0), 2)
      item = cv2.bitwise_not(img[y: y + h, x : x + w])

    # Ovo bi moglo pametnije al radi i ovako...  
    item = cv2.resize(item, (28, 28))
    # Pošaljemo modelu, vraća klasu 0, 1, 2...
    pred_index = model.predict_classes(np.reshape(item, [1, 28, 28, 1]))
    label = labels[pred_index[0]]
    # Dohvatimo labelu i ispišemo je
    cv2.putText(solution, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Cuvamo resenje u izlazni fajl
cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)
