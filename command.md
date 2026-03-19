expand prosedur 
```powershell
python3 data/dataset/expand_prosedur.py
```

with path
```powershell
python3 data/dataset/expand_prosedur.py --input /path/in.csv --output /path/out.csv
```
rebuild docker image
```powershell
# Bersihkan semua cache Docker dulu
docker system prune -f
docker compose down --rmi all

# Rebuild dengan progress detail
docker compose build --no-cache --progress=plain api 2>&1 | tee build.log

# Kalau sukses, jalankan
docker compose up -d

# Monitor startup (model loading butuh waktu ~60-90 detik)
docker logs -f fira-bot-api

# Build kedua image dulu
docker compose build --no-cache

# Baru jalankan
docker compose up -d

# push dockerhub
chmod +x push_to_dockerhub.sh
./push_to_dockerhub.sh

```

