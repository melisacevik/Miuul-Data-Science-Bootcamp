#sanal ortamların listelenmesi:
# conda env list

#sanal ortam oluşturma:
#conda create -n myenv

#sanal ortamı aktif etme:
#conda activate myenv

#Yüklü paketlerin listelenmesi:
#conda list

#Paket yükleme:
# conda install numpy

#Aynı anda birden fazla paket yükleme:
# conda install numpy scipy pandas

#Paket silme:
# conda remove package_name ( conda remove numpy gibi.)

#Belirli bir versiyona göre paket yükleme:
# conda install numpy=1.20.1    // pip'te == kullanılır.

# Paket yükseltme: ( numpy'ın en güncel haline gitmesini bekleriz )
# conda upgrade conda

# Tüm paketlerin yükseltilmesi:
# conda upgrade --all

# Pip açılımı nedir ?
# pip: pypi ( python package index ) paket yönetim aracı

# Paket yükleme:
# pip install package_name => pip install pandas

#Versiyon göre paket yükleme
# pip install pandas==2.1.1

# Oluşturduğumuz paketleri dosya şeklinde göndermek istersek
# conda env export > environment.yaml
# kontrolü => ls

# Başka bir iş ortamına , başka pc'ye geçtik, birine aktarmam gerekiyor.
# Bu çıktıyı kullanarak kullanarak nasıl bir çalışma ortamı yaratılabilir? Elimde var olan sanal ortam bilgi setini, ismini,
# paketlerini , paketlerin versiyonlarını barındıran bir dosyayı kullanarak sıfırdan aynı environment'i ayağa kaldırmak istiyorum.

# conda env create -f environment.yaml > conda list

# daha önce açılan environmenti kapatmak için:  conda deactivate

#daha önce açtığım ortamı silmek için : conda env remove -n myenv > conda env list >
#hazır olan ortamı çalıştırmak icin conda env create -f environment.yaml > conda activate myenv

# Artık python programını kullanırken kendi sanal ortamımızı oluşturabilir, paket yönetimi yapabileceğiz.