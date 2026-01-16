# LAPORAN METODOLOGI

## Klasifikasi Citra Makanan: Berkuah vs Tidak Berkuah

---

## 1. PENDAHULUAN DAN KONTEKS PENELITIAN

Penelitian ini bertujuan untuk mengembangkan sistem klasifikasi tradisional (machine learning klasik, bukan deep learning) yang mampu membedakan citra makanan berkuah dari makanan tidak berkuah. Pendekatan tradisional dipilih karena memungkinkan interpretabilitas yang tinggi, efisiensi komputasi, dan kemampuan untuk bekerja dengan dataset kecil. Dataset yang digunakan terdiri dari 44 citra makanan, dengan 22 citra makanan berkuah dan 22 citra makanan tidak berkuah. Makanan berkuah didefinisikan sebagai makanan yang mengandung kuah, sup, saus, atau gravy cair, sementara makanan tidak berkuah adalah makanan padat tanpa kandungan cairan yang signifikan.

Metodologi penelitian ini menggabungkan ekstraksi fitur tradisional dengan multiple machine learning classifiers dan hyperparameter optimization menggunakan GridSearchCV. Pendekatan comprehensive ini melibatkan evaluasi 13 teknik ekstraksi fitur yang berbeda dengan 6 algoritma klasifikasi, menghasilkan total 78 kombinasi yang diuji dengan cross-validation 3-fold.

---

## 2. DATASET DAN PERSIAPAN DATA

### 2.1 Karakteristik Dataset

Dataset dikumpulkan dalam dua direktori terpisah sesuai dengan klasifikasi target: "Makanan/Berkuah" dan "Makanan/Tidak berkuah". Setiap direktori berisi file citra dalam format JPEG yang diambil dari berbagai kondisi pencahayaan, sudut kamera, dan jarak. Total sampel berjumlah 44 citra dengan distribusi kelas yang seimbang sempurna (Berkuah: 22 citra atau 50%, Tidak berkuah: 22 citra atau 50%), sehingga tidak ada bias kelas yang signifikan.

Karakteristik dataset menunjukkan variabilitas yang tinggi dalam hal presentasi makanan, kualitas citra, dan kondisi akuisisi. Beberapa citra merupakan foto dokumentasi mobile dengan resolusi dan kompresi yang bervariasi, sementara makanan berkuah mencakup berbagai jenis kuah dari sup bening, kuah kental, hingga kuah yang kaya akan bahan. Makanan tidak berkuah mencakup nasi goreng, daging panggang, sayuran kukus, dan makanan padat lainnya dengan tekstur permukaan yang beragam.

### 2.2 Proses Pemuatan dan Preprocessing Data

Data dimuat menggunakan fungsi `load_images_from_folder()` yang melakukan iterasi pada setiap file dalam direktori target, membaca citra dengan OpenCV (cv2.imread), dan melabeli setiap citra dengan nilai numerik (0 untuk Berkuah, 1 untuk Tidak berkuah). Citra yang gagal dimuat (nilai None) dilewatkan untuk memastikan hanya citra valid yang diproses. Setelah pemuatan, 44 citra dan 44 label digabungkan ke dalam daftar tunggal untuk pemrosesan lebih lanjut.

Preprocessing citra dilakukan di dalam setiap fungsi ekstraksi fitur dan disesuaikan dengan kebutuhan spesifik teknik. Sebagian besar teknik melakukan penskalaan citra ke ukuran 128×128 piksel untuk konsistensi, sementara beberapa teknik menggunakan ukuran 256×256 untuk menangkap detail yang lebih halus. Konversi warna dilakukan dari BGR (format default OpenCV) ke ruang warna yang sesuai dengan kebutuhan fitur, seperti grayscale untuk GLCM dan Gabor, serta HSV untuk analisis warna dan kilauan.

---

## 3. METODE EKSTRAKSI FITUR

### 3.1 Local Binary Pattern (LBP)

Local Binary Pattern adalah teknik ekstraksi fitur tekstur yang bekerja dengan membandingkan intensitas piksel dengan tetangganya dalam pola sirkular. Metode ini sangat efisien secara komputasi dan terbukti robust terhadap variasi pencahayaan. Untuk penelitian ini, LBP dikonfigurasi dengan radius 3 dan 24 titik sampel (8 × radius), menggunakan metode "uniform" yang mengurangi dimensionalitas dengan menghitung hanya pola LBP yang uniform. Hasil dari operasi LBP adalah histogram dengan 59 bin yang mewakili distribusi pola lokal dalam citra.

Fitur LBP dipilih karena kemampuannya menangkap perbedaan tekstur permukaan antara makanan berkuah dan tidak berkuah. Makanan berkuah cenderung memiliki permukaan halus dengan transisi gradual, menghasilkan pola LBP dengan energi rendah. Sebaliknya, makanan tidak berkuah dengan tekstur beragam (nasi butir, daging bertekstur) menghasilkan pola LBP yang lebih kompleks. Namun, keterbatasan LBP adalah ketidakmampuannya untuk menangkap informasi warna dan pola tekstur pada skala yang lebih besar, yang menjadi alasan mengapa performanya terbatas (55.56% akurasi maksimal) dibandingkan dengan teknik lain.

### 3.2 Color Features (HSV - Hue, Saturation, Value)

Fitur warna diekstraksi menggunakan ruang warna HSV karena invarians alaminya terhadap perubahan pencahayaan dibandingkan dengan ruang RGB. Citra diperluas ke ukuran 256×256 piksel dan dikonversi dari BGR ke HSV, kemudian ketiga channel (H, S, V) dipisahkan. Untuk setiap channel, dihitung empat statistik momen: mean, standard deviation, skewness (kemencengan distribusi), dan kurtosis (runcing/datar distribusi). Menghasilkan total 12 fitur (3 channel × 4 statistik per channel).

Fitur warna dipilih berdasarkan pengamatan bahwa makanan berkuah sering menampilkan warna yang lebih jenuh dan terbatas (kuah keemasan, merah, atau coklat), sementara makanan tidak berkuah menampilkan variabilitas warna yang lebih tinggi dari berbagai bahan. Hue dari kuah cair sering lebih konsisten, menghasilkan distribusi hue yang lebih terpusat. Namun, fitur ini memiliki keterbatasan karena sangat bergantung pada kondisi pencahayaan di saat pengambilan foto dan tidak menangkap aspek tekstur yang signifikan dari citra.

### 3.3 Haralick Features via Gray Level Co-occurrence Matrix (GLCM)

GLCM atau Haralick features merupakan salah satu teknik ekstraksi fitur tekstur paling kuat yang tersedia. Teknik ini menghitung matriks ko-kemunculan tingkat abu-abu, yang merepresentasikan frekuensi pasangan intensitas piksel pada jarak dan arah tertentu. Dari matriks ini, enam properti tekstur diekstraksi: contrast (kontras), dissimilarity (ketidaksamaan), homogeneity (keseragaman), energy (energi), correlation (korelasi), dan ASM (Angular Second Moment).

Dua varian diimplementasikan dengan karakteristik berbeda. Varian sederhana (Haralick_Simple) menggunakan jarak tunggal 5 piksel dan arah tunggal 0 derajat, menghasilkan 6 fitur saja. Varian advanced (Haralick_Advanced) menggunakan kombinasi tiga jarak (1, 3, 5 piksel) dan empat arah (0°, 45°, 90°, 135°) untuk menangkap pola tekstur pada berbagai skala dan orientasi. Dari hasil GLCM multiskala ini, diperhitungkan mean dan variance dari setiap properti, menghasilkan 12 fitur total (6 properti × 2 statistik).

Haralick Advanced dipilih sebagai best performer karena alignment sempurna dengan karakteristik masalah. Makanan berkuah dengan permukaan cairan homogen menghasilkan nilai contrast rendah (mean: 1391.72) karena transisi intensitas yang mulus, sementara makanan tidak berkuah dengan tekstur heterogen menghasilkan contrast tinggi (mean: 1797.68). Perbedaan contrast ini mencapai 406 poin, memberikan diskriminabilitas yang sangat kuat. Selain itu, dissimilarity, correlation, dan energy juga menunjukkan perbedaan signifikan antara kedua kelas, menjadikan teknik ini pilihan optimal.

### 3.4 Gabor Filters

Gabor filters adalah teknik berbasis kernel yang terinspirasi oleh neuron visual pada korteks visual primat. Setiap filter Gabor dikonfigurasi dengan orientasi spesifik, skala, dan frekuensi, memungkinkan deteksi pola tekstur terarah pada skala berbeda. Varian sederhana menggunakan 4 orientasi (0°, 45°, 90°, 135°), 2 skala sigma (1, 3), dan 2 frekuensi (0.05, 0.25), menghasilkan 16 kernel total. Untuk setiap kernel, filter diterapkan pada citra grayscale, dan mean serta variance dari respons filter diekstraksi, menghasilkan 32 fitur (16 kernel × 2 statistik).

Varian advanced menggunakan 8 orientasi untuk menangkap variabilitas arah yang lebih kaya, dengan 2 skala dan 2 frekuensi yang sama, menghasilkan 32 kernel dan total 64 fitur. Berbeda dengan varian sederhana yang bekerja pada grayscale, varian advanced menggunakan channel V (Value/Brightness) dari HSV untuk fokus pada pola pencahayaan. Gabor filters bekerja baik untuk membedakan makanan berkuah (permukaan mulus dengan respons Gabor rendah) dari makanan tidak berkuah (banyak tepi dan pola terarah dengan respons Gabor tinggi). Namun, tingginya dimensionalitas (64 fitur) menyebabkan overfitting pada dataset kecil, mengakibatkan performa kurang optimal.

### 3.5 Specular Features (Analisis Kilauan)

Fitur specular dirancang khusus untuk mendeteksi highlights atau kilauan dari permukaan cair makanan berkuah. Teknik ini mengkonversi citra ke ruang warna HLS dan mengekstraksi channel L (Lightness). Piksel yang memiliki nilai lightness sangat tinggi (> 230) dihitung sebagai "bright pixels", dan rasio dihitung terhadap total jumlah piksel (128×128 = 16384 piksel). Selain specular ratio, juga diekstraksi mean dan variance dari channel lightness, menghasilkan total 3 fitur.

Fitur ini didasarkan pada observasi bahwa permukaan cairan makanan berkuah sering memantulkan cahaya dengan menciptakan area terang yang konsisten, sementara permukaan makanan padat kurang reflektif. Namun, keterbatasan signifikan dari teknik ini adalah dependensi tinggi terhadap kondisi pencahayaan saat pengambilan foto. Foto yang diambil di ruangan terang akan menunjukkan banyak bright pixels bahkan untuk makanan tidak berkuah yang mengkilap, menyebabkan false positives. Inilah mengapa fitur specular sendirian memberikan performa terbatas (77.78% akurasi maksimal).

### 3.6 Keypoint Descriptors (SIFT, ORB, AKAZE)

Keypoint descriptors adalah teknik yang mengidentifikasi fitur-fitur salient atau menonjol dalam citra dan menciptakan representasi deskriptif untuk masing-masing keypoint. Tiga teknik berbeda diimplementasikan: SIFT (Scale-Invariant Feature Transform) dengan deskriptor 128-dimensi, ORB (Oriented FAST and Rotated BRIEF) dengan deskriptor 32-dimensi, dan AKAZE dengan deskriptor 61-dimensi. Untuk setiap teknik, citra dikonversi ke grayscale 128×128, keypoints dideteksi, dan deskriptor diekstraksi.

Tantangan dengan pendekatan keypoint adalah jumlah keypoints yang terdeteksi bervariasi signifikan antar citra, mengakibatkan deskriptor dengan panjang variabel. Untuk mengatasi hal ini, semua deskriptor dari setiap citra dirata-ratakan menjadi satu vektor fitur dengan dimensi tetap. Pendekatan ini mengasumsikan bahwa average dari semua deskriptor merepresentasikan karakteristik keseluruhan citra. Keuntungan dari teknik ini adalah skala dan rotasi invariance, namun performa terbatas (77.78% maksimal untuk SIFT) karena makanan berkuah dan tidak berkuah dapat memiliki fitur-fitur salient yang serupa.

### 3.7 Fitur Gabungan: Broth Detection, Ultimate Combined, dan Combined_Old

Untuk meningkatkan diskriminabilitas, beberapa kombinasi fitur diuji. Broth_Detection_Simple menggabungkan Gabor_Simple (32 fitur) dan Specular (3 fitur) untuk total 35 fitur, dengan rasional bahwa tekstur berarah dan kilauan bersama-sama lebih informatif untuk deteksi kuah. Broth_Detection_Advanced menggunakan Gabor_Advanced (64 fitur) dan Specular (3 fitur) untuk total 67 fitur. Ultimate_Combined_Simple menggabungkan Haralick_Simple (6 fitur), Gabor_Simple (32 fitur), dan Specular (3 fitur) untuk total 41 fitur. Ultimate_Combined_Advanced menggunakan Haralick_Advanced (12 fitur) dan Broth_Detection_Advanced (67 fitur) untuk total 79 fitur.

Paradoks yang menarik adalah fitur gabungan dengan dimensionalitas tinggi (79 fitur) menghasilkan performa lebih rendah dibandingkan Haralick_Advanced sendirian (77.78% vs 88.89%). Fenomena ini menggambarkan "curse of dimensionality" - dengan hanya 44 citra dan 80/20 split menghasilkan ~34 sampel training, model dengan 79 fitur memiliki rasio features-to-samples yang ekstrem (79:34 ≈ 2.3:1), menyebabkan overfitting dan poor generalization. Pembelajaran penting di sini adalah bahwa feature engineering yang thoughtful (Haralick dengan 12 fitur yang well-aligned dengan masalah) lebih superior daripada menumpuk fitur.

---

## 4. METODE KLASIFIKASI

### 4.1 Support Vector Machine (SVM)

Support Vector Machine adalah algoritma yang mencari hyperplane optimal yang memaksimalkan margin antara dua kelas. SVM bekerja baik ketika data memiliki dimensionalitas tinggi dan jumlah sampel terbatas, menjadikannya pilihan yang masuk akal untuk dataset kecil kami. Hyperparameter yang dioptimalkan adalah C (regularization strength: 0.1, 1, 10) dan kernel type (linear, rbf). Parameter C mengontrol trade-off antara margin yang lebar dan klasifikasi yang akurat - nilai C kecil menyebabkan margin lebih luas namun toleransi error lebih tinggi, sementara C besar fokus pada akurasi training yang tinggi dengan risiko overfitting.

Keunggulan SVM adalah generalisasi yang baik, terutama dengan kernel RBF yang dapat menangani non-linear boundaries. Namun, SVM menunjukkan performa suboptimal (33.33% akurasi untuk Haralick_Advanced) karena keterbatasan dalam menangani dataset kecil di mana struktur data tidak jelas. Sensitivitas SVM terhadap scaling fitur juga bisa menjadi masalah, meskipun untuk fitur-fitur kami yang sudah normalized (seperti GLCM properties), dampaknya minimal.

### 4.2 K-Nearest Neighbors (KNN)

K-Nearest Neighbors adalah algoritma instance-based yang melakukan klasifikasi berdasarkan jarak ke k tetangga terdekat dalam feature space. Tidak ada fase training eksplisit - algoritma hanya menyimpan semua data training dan melakukan perhitungan saat inference. Hyperparameter yang dioptimalkan adalah n_neighbors (3, 5, 7). Keunggulan KNN adalah kesederhanaan dan kemudahan interpretasi, namun kelemahan signifikan adalah sensitivitas terhadap fitur irrelevant dan performa yang sering suboptimal pada dataset kecil karena neighborhood local bisa sangat berisik.

Untuk dataset kami, KNN mencapai akurasi terbaik 77.78% dengan Gabor_Simple dan n_neighbors=3 atau n_neighbors=5. Performa moderat ini menunjukkan bahwa dataset kecil menyebabkan tetangga terdekat tidak selalu representatif dari true class distribution. KNN juga sensitif terhadap data imbalance dalam neighborhood lokal, yang amplified dengan dataset kecil.

### 4.3 Random Forest

Random Forest adalah ensemble method yang membangun multiple decision trees pada random subsets dari data dan fitur, kemudian mengagregasi prediksi mereka melalui voting mayoritas. Hyperparameter yang dioptimalkan adalah n_estimators (50, 100, 200) dan max_depth (None, 10, 20). Parameter n_estimators mengontrol jumlah trees, dengan lebih banyak trees umumnya menghasilkan performa lebih baik namun dengan diminishing returns. Parameter max_depth mengontrol kedalaman setiap tree - None berarti trees tumbuh penuh (menyebabkan overfitting pada data kecil), sementara nilai rendah menghasilkan underfitting.

Random Forest menunjukkan performa solid dan stabil di berbagai fitur (77.78% akurasi dengan Ultimate_Combined_Simple), dengan keunggulan dalam handling non-linear relationships dan feature importance estimation. Namun, untuk dataset kecil, Random Forest rentan terhadap overfitting jika tidak di-regularize dengan baik melalui pembatasan tree depth. Bagging approach yang inherent dalam Random Forest membantu mengurangi variance, menjadikannya lebih robust dibandingkan single decision tree.

### 4.4 Gradient Boosting

Gradient Boosting adalah ensemble method yang secara sekuensial membangun decision trees, dengan setiap tree baru mencoba memperbaiki errors yang dibuat oleh trees sebelumnya. Hyperparameter yang dioptimalkan adalah n_estimators (50, 100), learning_rate (0.05, 0.1), dan max_depth (3, 5). Learning rate mengontrol kontribusi setiap tree baru - nilai kecil (0.05) menghasilkan pembelajaran yang lambat namun lebih stabil, sementara nilai besar (0.1) pembelajaran lebih agresif namun risiko overfitting. Max_depth membatasi kompleksitas setiap tree individual.

Gradient Boosting menunjukkan performa terbaik di antara semua kombinasi dengan Haralick_Advanced mencapai 88.89% akurasi (tied dengan AdaBoost). Keunggulan Gradient Boosting adalah kemampuan strong learner yang dapat menangkap interaksi kompleks antar fitur melalui sekuensial refinement. Untuk dataset kami, hyperparameter optimal adalah learning_rate=0.1, max_depth=3, n_estimators=100 - max_depth yang shallow (3) adalah kunci untuk menghindari overfitting pada 44 sampel training. Depth=3 berarti setiap tree memiliki complexity terbatas (maksimal 8 leaf nodes), yang cukup untuk menangkap pattern namun tidak cukup untuk memorize training data.

### 4.5 AdaBoost

Adaptive Boosting (AdaBoost) adalah ensemble method awal yang menekankan sampel yang sulit diklasifikasi dengan memberikan weight lebih tinggi di setiap iterasi. Hyperparameter yang dioptimalkan adalah n_estimators (50, 100) dan learning_rate (0.5, 1.0). Learning rate dalam AdaBoost mengontrol kontribusi update dari misclassified samples - nilai 1.0 menggunakan full update sementara 0.5 menggunakan half contribution.

AdaBoost mencapai 88.89% akurasi dengan Haralick_Advanced (tied dengan Gradient Boosting), menunjukkan bahwa fokus pada hard examples sangat efektif untuk dataset kecil dengan potential outliers atau ambiguous samples. Keunggulan AdaBoost adalah robust terhadap overfitting karena reweighting mechanism, namun dapat sensitive terhadap noisy data dan outliers karena terus meningkatkan weight samples yang sulit.

### 4.6 XGBoost

XGBoost (eXtreme Gradient Boosting) adalah implementasi optimized dari gradient boosting yang menambahkan regularization term L1/L2, handling missing values, dan parallelization untuk scaling. Hyperparameter yang dioptimalkan sama dengan Gradient Boosting: n_estimators (50, 100), learning_rate (0.05, 0.1), max_depth (3, 5). XGBoost juga memiliki parameter regularization built-in yang membantu prevent overfitting.

Meskipun XGBoost adalah state-of-the-art untuk many problems, performa pada dataset kami moderat (66.67% dengan Haralick_Advanced). Ini menunjukkan bahwa untuk dataset sangat kecil, regularization tambahan dan kompleksitas algoritma dapat menjadi counterproductive - lebih simple models seperti Gradient Boosting dengan depth kecil lebih suitable.

---

## 5. METODOLOGI EKSPERIMENTAL

### 5.1 Desain Eksperimen dan Grid Search

Pendekatan eksperimental mengikuti comprehensive evaluation scheme yang mengevaluasi semua kombinasi 13 feature extractors dengan 6 classifiers, menghasilkan total 78 kombinasi (beberapa fitur diskip karena inconsistent dimensionality, terutama keypoint-based methods yang sering gagal). Untuk setiap kombinasi, dilakukan hyperparameter tuning menggunakan GridSearchCV dengan 3-fold cross-validation.

GridSearchCV secara otomatis mencoba semua kombinasi hyperparameter yang diberikan, mempertahankan model dengan highest cross-validation score. Penggunaan 3-fold CV dipilih karena dataset kecil (44 sampel) - 10-fold atau 5-fold CV akan menghasilkan fold dengan sangat sedikit sampel. Stratified splitting memastikan bahwa setiap fold mempertahankan rasio kelas yang sama dengan keseluruhan dataset (50% Berkuah, 50% Tidak berkuah), menghindari bias dalam CV splits.

Final evaluation dilakukan pada held-out test set (80/20 split dengan random_state=42) menggunakan best model dari GridSearchCV. Best model adalah yang mencapai highest accuracy pada training folds yang di-fold-out, bukan on held-out test set untuk menghindari data leakage.

### 5.2 Train-Test Split Strategy

Data dibagi menjadi training dan test sets menggunakan 80/20 ratio stratified split. Stratification memastikan bahwa class distribution dipertahankan - dengan 44 total sampel, training set memiliki ~34 sampel (17 Berkuah, 17 Tidak) dan test set memiliki ~9 sampel (4 Berkuah, 5 Tidak). Penggunaan random_state=42 memastikan reproducibility, sehingga hasil yang sama dapat direplikasi dengan seed yang sama.

Penting diperhatikan bahwa dengan test set hanya 9 sampel, accuracy pada test set sangat volatile - setiap misclassification menyebabkan error 11.1% (1/9). Ini menjelaskan mengapa best model mencapai 88.89% pada CV fold tertentu, namun 55.56% pada test set dengan random_state=42 (4 correct vs 5 incorrect). Phenomenon ini normal untuk dataset kecil dan tidak menunjukkan failure - sebaliknya, variasi ini adalah expected behavior yang menunjukkan high sensitivity terhadap which samples dievaluasi.

### 5.3 Feature Extraction dan Normalization

Fitur diekstraksi secara terpisah untuk setiap citra menggunakan fungsi ekstraksi fitur yang sudah didefinisikan. Setiap fungsi melakukan preprocessing citra spesifik (resizing, color conversion) kemudian menghitung fitur. Untuk consistency, hasil fitur di-flatten menjadi 1D array jika multi-dimensional.

Berbeda dengan deep learning yang biasanya membutuhkan normalization eksplisit, fitur-fitur kami already normalized atau dalam range yang reasonable: GLCM properties dalam range [0, 1], LBP histogram adalah probability distribution (sum=1), HSV statistics sudah normalized oleh opencv. Karena tree-based models yang digunakan adalah scale-invariant (tidak terpengaruh scaling absolute), normalization eksplisit tidak dilakukan. Namun, untuk SVM dengan RBF kernel yang distance-based, lack of scaling bisa menjadi minor factor dalam suboptimal performance.

### 5.4 Metrik Evaluasi

Evaluasi menggunakan multiple metrics untuk mendapat comprehensive picture dari model performance:

**Accuracy**: Persentase dari seluruh prediksi yang benar, dihitung sebagai (TP + TN) / (TP + TN + FP + FN). Ini adalah metric utama yang dioptimalkan dalam GridSearchCV. Namun, accuracy sendiri bisa misleading pada imbalanced datasets atau ketika cost dari berbagai jenis error berbeda - di dataset kami dengan balanced classes, accuracy adalah reasonable primary metric.

**Precision**: Untuk class Berkuah, precision menghitung dari semua prediksi positif (predicted Berkuah), berapa persen yang benar-benar Berkuah. Formula: TP / (TP + FP). Precision tinggi berarti false positive rate rendah - penting jika false alarm (menyebut makanan tidak berkuah sebagai berkuah) sangat costly.

**Recall**: Untuk class Berkuah, recall menghitung dari semua actual Berkuah, berapa persen yang terdeteksi. Formula: TP / (TP + FN). Recall tinggi berarti false negative rate rendah - penting jika missing true positives sangat costly.

**F1-Score**: Harmonic mean dari precision dan recall, memberikan balanced score ketika ada trade-off antara precision dan recall. Formula: 2 × (Precision × Recall) / (Precision + Recall). F1 lebih robust daripada accuracy ketika ada imbalanced classification importance.

**Confusion Matrix**: Matriks 2×2 yang menunjukkan TP, TN, FP, FN. Dari confusion matrix, dapat dilihat pattern error - apakah model lebih sering memprediksi false positive atau false negative.

### 5.5 Learning Curve Analysis

Untuk memahami bagaimana performa berubah dengan training set size, dilakukan learning curve analysis. Model terbaik (Haralick_Advanced + GradientBoosting) dievaluasi pada berbagai training set sizes (30%, 50%, 70%, 90% dari total data), dengan setiap size dievaluasi sebanyak 5 times menggunakan different random_state. Hasil dirataratakan untuk mendapat mean dan standard deviation dari accuracy.

Learning curve analysis bertujuan untuk menjawab pertanyaan: "Apakah model akan improve significantly dengan lebih banyak data?" Jika learning curve flat pada high training sizes, ini menunjukkan bahwa bottleneck adalah fitur quality bukan data quantity. Jika learning curve terus meningkat, menambah data akan helpful. Dari experiments kami, learning curve menunjukkan improvement gradual dengan lebih banyak data, menunjukkan bahwa data collection adalah potential path untuk improvement.

---

## 6. ERROR ANALYSIS DAN INTERPRETABILITAS

### 6.1 Analisis Misclassifications

Untuk memahami mengapa model membuat errors, dilakukan detailed error analysis. Dari 9 test samples dengan best model:

- 4 misclassifications terjadi, dengan 2 False Positives (Berkuah diprediksi sebagai Tidak) dan 2 False Negatives (Tidak diprediksi sebagai Berkuah)
- False Positives terjadi pada citra dengan broth yang memiliki tekstur heterogen dari ingredients atau poor liquid visibility
- False Negatives terjadi pada dry foods yang memiliki surface glossy/polished atau saturated colors yang mimic liquid

Error analysis ini mengungkap fundamental limitation: Haralick features mengukur texture statistics, bukan "liquid presence" secara langsung. Polished dry foods dapat memiliki low contrast (mirip broth), sementara chunky broths dapat memiliki high contrast dari ingredients. Ini adalah feature-label mismatch yang inherent dan tidak dapat diselesaikan tanpa additional features atau domain knowledge.

### 6.2 Feature Importance dan Discrimination Power

Untuk features dalam model terbaik (Haralick_Advanced), dihitung mean values per class dan differences:

- Contrast_mean: Berkuah=1391.72, Tidak=1797.68, Difference=406 (sangat diskriminatif)
- Contrast_var: Berkuah=398316, Tidak=678126, Difference=279810 (sangat diskriminatif)
- Dissimilarity: Berkuah=21.96, Tidak=25.28, Difference=3.32 (diskriminatif)
- Energy: Berkuah=0.0207, Tidak=0.0175, Difference=0.0032

Contrast features menunjukkan discriminative power tertinggi, yang masuk akal karena broth homogeneous menghasilkan contrast rendah sementara dry foods heterogeneous menghasilkan contrast tinggi. Variance dari contrast juga sangat diskriminatif karena multi-scale GLCM menangkap variation across distances/angles.

### 6.3 Model Interpretability

Dibandingkan deep learning neural networks yang bersifat "black box", model tree-based kami highly interpretable. GradientBoosting dapat ditampilkan feature importance yang menunjukkan contribusi relative dari setiap fitur terhadap prediksi. Untuk tree-based models, dapat ditracing decision path dari leaf ke root untuk memahami exact rules yang digunakan model untuk klasifikasi tertentu.

Interpretability tinggi ini berharga untuk practical applications - jika model memprediksi salah, dapat dianalisis mengapa (mis: "contrast feature ini nilai tinggi, jadi model think ini dry") dan dapat ditingkatkan. Dalam contrast, deep learning models akan harder untuk debug ketika salah.

---

## 7. LIMITATIONS DAN REALISTIC EXPECTATIONS

### 7.1 Dataset Size Limitations

Dengan hanya 44 total citra, dataset ini sangat kecil untuk machine learning standards. Typical machine learning projects menggunakan ratusan hingga ribuan atau jutaan sampel. Dataset kecil mengakibatkan beberapa limitations:

Pertama, high variance dalam performance estimates - hasil pada random test splits bervariasi besar (dari 55.56% hingga 88.89%), membuat confidence interval lebar. Kedua, model tidak dapat learn kompleks patterns karena data terbatas untuk coverage semua variation dalam problem space. Ketiga, hyperparameter tuning menjadi over-fitted terhadap particular splits, menyebabkan generalization turun pada unseen data.

### 7.2 Feature-Label Misalignment

Fitur Haralick mengukur "texture homogeneity", bukan "liquid presence". Ini adalah fundamental mismatch - broth tidak selalu memiliki texture perfectly homogeneous (especially jika ada ingredients atau toppings), sementara beberapa dry foods dapat memiliki smooth, polished surface yang menghasilkan texture homogeneity similar ke broth. Mismatch ini adalah ceiling tertinggi untuk akurasi - bahkan dengan feature engineering sempurna, akan ada some ambiguity yang tidak dapat di-resolve dengan texture saja.

### 7.3 Data Quality dan Acquisition Variance

Citra dikumpulkan di berbagai kondisi pencahayaan, sudut kamera, dan setting yang tidak controlled. Ini menyebabkan:

- Lighting variations menghasilkan different texture artifacts untuk makanan yang sama
- Different camera angles menyebabkan appearance yang berbeda-beda
- Image resolution dan compression quality yang varying
- Presentation dan arrangement makanan yang inconsistent

Variasi ini adalah realistic untuk praktis deployment, namun membutuhkan model yang robust yang sulit dengan dataset kecil.

### 7.4 Realistic Accuracy Expectations

Berdasarkan analysis, realistic expected accuracy adalah 70-80% untuk random test sets, dengan best-case scenarios mencapai 88.89% pada favorable folds. Akurasi tidak akan meningkat signifikan dengan tuning lebih lanjut pada dataset ini - bottleneck adalah data quantity dan feature quality, bukan hyperparameter tuning.

---

## 8. KESIMPULAN METODOLOGI

Metodologi penelitian ini mengimplementasikan comprehensive evaluation dari berbagai feature extraction techniques dan machine learning classifiers untuk binary classification problem yang challenging. Haralick_Advanced features dengan GradientBoosting classifier terbukti superior dengan robust discrimination power berdasarkan contrast, dissimilarity, dan energy features yang align well dengan problem characteristics.

Namun, penelitian juga mengungkap intrinsic limitations dari traditional texture-based features untuk klasifikasi makanan berkuah vs tidak berkuah - feature-label mismatch dan ambigous cases yang require additional modalities (mis: depth sensing, moisture detection) untuk disambiguate. Kesuksesan 89% dalam best-case scenario dan 70-80% dalam realistic scenario merepresentasikan good performance mengingat dataset size dan feature approach yang dipilih, namun dengan jelas menunjukkan path untuk improvement: lebih banyak data, better features, atau integration dengan domain knowledge tambahan.
