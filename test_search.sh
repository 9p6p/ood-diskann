num_threads=1
topk=1

cd build

data=t2i-10M
prefix=/home/yuxiang/RoarGraph-tochange/datasets/${data}
saveprefix=/home/yuxiang/RoarGraph-tochange/indices/${data}
rootsave=/home/yuxiang/RoarGraph-tochange/results/${data}
graph=robustvamana.index

./tests/search_memory_index_dynamic --data_type float \
--dist ip \
--index_save_path ${saveprefix}/${graph} \
--gt_path ${prefix}/gt.10k.ibin \
--query_path ${prefix}/query.10k.fbin \
--search_list 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 160 170 180 190 200 220 240 260 280 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
--k ${topk} \
--num_threads ${num_threads} \
--evaluation_save_path ${rootsave}/test_${graph}_top${topk}_T${num_threads}.csv
# 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 