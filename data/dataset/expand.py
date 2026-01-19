# Define patterns for each intent
patterns_dict = {
    "ak1": [
        "bagaimana cara buat kartu kuning?",
        "syarat bikin ak1 di tegal",
        "daftar pencari kerja online",
        "link pendaftaran kartu kuning",
        "cetak kartu kuning mandiri",
        "langkah-langkah buat ak1",
        "kartu kuning disnaker tegal",
        "cara registrasi bursakerja jateng",
        "dimana bikin kartu kuning?",
        "apakah buat ak1 bisa online?",
        "persyaratan kartu kuning terbaru",
        "bikin kartu kuning gratis ga?",
        "cara dapat kartu kuning buat melamar kerja",
        "proses pembuatan ak1 berapa hari?",
        "kartu kuning online tegal"
    ],
    "nib": [
        "cara buat nib di tegal",
        "daftar oss rba online",
        "syarat nomor induk berusaha",
        "bikin nib gratis dimana?",
        "link oss go id",
        "daftar nib mandiri",
        "legalitas usaha nib tegal",
        "syarat utama buat nib",
        "cara dapat nomor induk berusaha",
        "apakah nib wajib buat umkm?",
        "dimana loket dpmptsp mpp?",
        "bikin izin usaha online tegal",
        "nib rba itu apa?",
        "prosedur daftar oss rba",
        "cara urus nib di mpp"
    ],
    "jam_buka_layanan": [
        "jam berapa pelayanan buka?",
        "hari sabtu buka tidak?",
        "jadwal operasional kantor",
        "jam pelayanan hari jumat",
        "apakah hari minggu buka?",
        "jam operasional mpp",
        "kantor tutup jam berapa?",
        "pelayanan buka jam berapa?",
        "jadwal buka hari senin",
        "info jam pelayanan publik",
        "hari libur nasional buka ga?",
        "kapan jam istirahat pelayanan?",
        "layanan buka jam brp?",
        "jadwal kerja kantor tegal",
        "jam tutup pelayanan"
    ],
    "itr_info": [
        "apa itu itr?",
        "cara urus informasi tata ruang",
        "syarat pbg imb tegal",
        "daftar sicantik dpupr",
        "cek tata ruang lahan",
        "dokumen perencanaan penggunaan lahan",
        "dimana loket dpupr?",
        "syarat utama itr tegal",
        "cara dapet sertifikat itr",
        "itr buat apa?",
        "prosedur tata ruang online",
        "biaya urus itr",
        "syarat legalitas lahan",
        "info tata ruang kabupaten tegal",
        "pengurusan izin lahan"
    ],
    "sls_info": [
        "apa itu sertifikat laik sehat?",
        "syarat sls dinkes",
        "cara urus izin laik sehat resto",
        "sertifikat higienis hotel",
        "laik sehat depot air minum",
        "syarat dinkes mpp",
        "biaya urus sls",
        "hasil uji lab air bersih sls",
        "apa saja syarat laik sehat?",
        "pendaftaran sls online",
        "sls wajib buat usaha apa?",
        "dimana urus sertifikat laik sehat?",
        "prosedur sanitasi dinkes",
        "rekomendasi phri buat sls",
        "uji lab bakteri dam"
    ],
    "nib_info": [
        "apa keuntungan punya nib?",
        "manfaat nomor induk berusaha",
        "kenapa harus punya nib?",
        "fungsi nib buat umkm",
        "apakah nib bisa buat modal bank?",
        "legalitas usaha terjamin nib",
        "nib pengganti tdp",
        "keuntungan daftar oss rba",
        "manfaat legalitas usaha",
        "nib buat pengadaan barang",
        "apa guna nib bagi pedagang?",
        "nib bisa buat dapat hibah?",
        "syarat kur pakai nib",
        "identitas tunggal pelaku usaha",
        "pentingnya nib"
    ],
    "lkpm_info": [
        "apa itu lkpm?",
        "cara lapor lkpm di oss",
        "pelaporan kegiatan penanaman modal",
        "kapan waktu lapor lkpm?",
        "laporan investasi triwulan",
        "cara isi data lkpm",
        "dimana lapor lkpm online?",
        "apakah lkpm wajib?",
        "sanksi tidak lapor lkpm",
        "tutorial lapor lkpm oss rba",
        "periode lapor lkpm semester",
        "lkpm tahap konstruksi",
        "laporan modal usaha",
        "login oss buat lkpm",
        "kendala lapor lkpm"
    ],
    "akta_lahir_info": [
        "cara buat akta kelahiran",
        "syarat akta lahir anak",
        "daftar akta lahir online sipandu",
        "akta lahir gratis atau bayar?",
        "syarat sipandu akta kelahiran",
        "bikin akta bayi baru lahir",
        "berapa lama akta lahir jadi?",
        "cara cetak akta lahir mandiri",
        "dokumen buat akta lahir",
        "akta kelahiran hilang urusnya gimana?",
        "pendaftaran bayi di sipandu",
        "formulir f-2.01 akta lahir",
        "buku nikah buat akta lahir",
        "cara upload berkas akta lahir",
        "akta lahir tegal gratis"
    ],
    "mati_info": [
        "cara urus akta kematian",
        "syarat akta kematian sipandu",
        "lapor orang meninggal tegal",
        "akta kematian buat apa?",
        "dokumen akta kematian",
        "cara hapus data kk orang meninggal",
        "surat keterangan kematian desa",
        "bikin akta kematian online",
        "biaya akta kematian",
        "prosedur akta kematian 1 hari jadi",
        "akta kematian buat waris",
        "lapor kematian di sipandu",
        "syarat update kk kematian",
        "akta kematian asli jenazah",
        "urus akta mati gratis"
    ],
    "kk_info": [
        "cara buat kartu keluarga",
        "syarat tambah anggota kk",
        "daftar kk online sipandu",
        "ganti kk rusak atau hilang",
        "pecah kk dari orang tua",
        "syarat kartu keluarga baru nikah",
        "cara cetak kk mandiri hvs",
        "update data kk online",
        "formulir f-1.01 kartu keluarga",
        "kk tegal gratis ga?",
        "syarat ganti data pendidikan di kk",
        "cara daftar sipandu kartu keluarga",
        "berapa lama urus kk online?",
        "tambah anak di kartu keluarga",
        "legalisir kk dimana?"
    ],
    "surat_pindah_luar": [
        "cara urus surat pindah keluar",
        "syarat pindah domisili dari tegal",
        "daftar skpwni online",
        "pindah keluar kabupaten tegal",
        "cabut berkas kk tegal",
        "syarat surat pindah sipandu",
        "formulir f-1.03 pindah",
        "urus surat pindah gratis",
        "cara cetak skpwni mandiri",
        "pindah antar provinsi urusnya gimana?",
        "berapa lama surat pindah jadi?",
        "prosedur cabut berkas domisili",
        "dokumen pindah penduduk",
        "pindah domisili luar daerah",
        "pindah ke luar kota tegal",
        "Bagaimana prosedur cabut berkas keluar kota dari Tegal?",
        "Saya mau pindah domisili ke luar daerah, apa saja yang harus disiapkan?",
        "Syarat mendapatkan surat SKPWNI untuk pindah antar provinsi.",
        "Cara mengurus surat pindah penduduk dari Kabupaten Tegal ke daerah lain.",
        "Langkah pendaftaran pindah keluar secara online melalui portal Sipandu."
    ],
    "loakk_info": [
        "apa itu layanan loakk?",
        "paket akta kk kia bayi",
        "cara daftar loakk sipandu",
        "layanan 3 in 1 bayi lahir",
        "loakk bayi baru lahir tegal",
        "syarat dapat akta kk kia sekaligus",
        "loakk gratis ga?",
        "kia bayi ambil dimana?",
        "proses loakk berapa hari?",
        "surat lahir rs buat loakk",
        "formulir loakk f-2.01",
        "syarat paket lahir 3 in 1",
        "loakk sipandu tegal",
        "cara urus akta dan kia sekaligus",
        "info layanan loakk"
    ],
    "sicantik_info": [
        "apa itu layanan sicantik?",
        "syarat sicantik cerai tegal",
        "update kk status cerai online",
        "sicantik integrasi pengadilan agama",
        "cara urus akta cerai ke kk",
        "formulir f-1.06 sicantik",
        "layanan cerai anti ribet tegal",
        "syarat update status cerai di kk",
        "sicantik disdukcapil tegal",
        "prosedur kk baru setelah cerai",
        "akta cerai pengadilan agama sipandu",
        "sicantik integrasi kk",
        "update data cerai gratis",
        "info pendaftaran sicantik",
        "integrasi data cerai"
    ],
    "ktp_info": [
        "cara buat ktp baru",
        "syarat rekam ktp tegal",
        "ktp hilang urusnya dimana?",
        "ganti foto ktp bisa ga?",
        "syarat ktp rusak sipandu",
        "rekam ktp di kecamatan atau capil?",
        "syarat buat ktp usia 17 tahun",
        "formulir f-1.21 ktp",
        "ktp tegal gratis atau bayar?",
        "cek stok blangko ktp",
        "cara urus ktp rusak online",
        "perekaman ktp el tegal",
        "syarat ganti status di ktp",
        "ktp el hilang surat kehilangan",
        "waktu jadi ktp berapa lama?"
    ]
}

import pandas as pd

# Read the null.csv file
df = pd.read_csv('null.csv')

# Expand the dataframe
expanded_rows = []
for index, row in df.iterrows():
    intent = row['intent']
    if intent in patterns_dict:
        for p in patterns_dict[intent]:
            new_row = row.copy()
            new_row['pattern'] = p
            expanded_rows.append(new_row)
    else:
        # If intent not in our hardcoded dict, keep original
        expanded_rows.append(row)

expanded_df = pd.DataFrame(expanded_rows)

# Save to CSV (overwrite null.csv)
expanded_df.to_csv('null.csv', index=False)

print(f"Expanded dataset size: {len(expanded_df)}")
print(expanded_df['intent'].value_counts())