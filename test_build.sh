mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

data=laion-10M

prefix=/home/yuxiang/RoarGraph-tochange/datasets/${data}
saveprefix=/home/yuxiang/RoarGraph-tochange/indices/${data}

./tests/build_with_query_data --data_type float \
--data_path ${prefix}/base.10M.fbin \
--query_file ${prefix}/query.train.10M.fbin \
--save_path ${saveprefix}/robustvamana.index \
--max_degree 64 \
--Lbuild 500 \
--alpha 1 \
--num_threads 64 \


# with match file, not disable   /root/autodl-tmp/datasets/t2i-10M/gt.train.dist.10M.fbin