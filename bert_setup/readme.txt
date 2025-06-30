ip: 140.118.164.51~54
password: aivc

ssh root@140.118.164.51 -p 10659

# 進入上傳的資料夾
cd /root/work_dir/bert_setup

# 賦予腳本執行權限
chmod +x install_env.sh setup_project.sh

# 1. 執行 Python 環境安裝 (只需要在帳號首次設定時執行一次)
bash ./install_env.sh

# 2. 執行 BERT 專案安裝
bash ./setup_project.sh