// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "index.h"
#include "memory_mapper.h"
#include "utils.h"

namespace po = boost::program_options;

template<typename T>
int search_memory_index(diskann::Metric& metric, const std::string& index_path,
                        const std::string& result_path_prefix,
                        const std::string& query_file,
                        std::string& truthset_file, const unsigned num_threads,
                        const unsigned               recall_at,
                        const std::vector<unsigned>& Lvec) {
  // Load the query file
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  // Check for ground truth
  bool calc_recall_flag = false;
  if (truthset_file != std::string("null") && file_exists(truthset_file)) {
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  } else {
    diskann::cout << " Truthset file " << truthset_file
                  << " not found. Not computing recall." << std::endl;
  }

  // Load the index
  // diskann::Index<T, uint32_t> index(metric, query_dim, 0, false);
  diskann::Index<T, uint32_t> index(metric, query_dim, 0, true, true, false);
  index.load(index_path.c_str(), num_threads,
             *(std::max_element(Lvec.begin(), Lvec.end())));
  std::cout << "Index loaded" << std::endl;
  if (metric == diskann::FAST_L2)
    index.optimize_index_layout();

  diskann::Parameters paras;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Avg dist cmps" << std::setw(20) << "Mean Latency (mus)"
            << std::setw(15) << "99.9 Latency";
  if (calc_recall_flag)
    std::cout << std::setw(12) << recall_string;
  std::cout << std::endl;
  std::cout << "==============================================================="
               "=================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  // uint32_t*     query_result_tags = new uint32_t[recall_at *
  // query_num*Lvec.size()];
  std::vector<std::vector<float>> query_result_dists(Lvec.size());
  std::vector<float>              latency_stats(query_num, 0);
  std::vector<unsigned>           cmp_stats(query_num, 0);

  // index.enable_delete();
  // index.disable_delete(paras, true);

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint32_t* query_result_tags = new uint32_t[recall_at * query_num];
    _u64      L = Lvec[test_id];
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    query_result_ids[test_id].resize(recall_at * query_num);

    // std::vector<uint32_t> tags(index.get_num_points());
    // std::iota(tags.begin(), tags.end(), 0);

    std::vector<T*> res = std::vector<T*>();

    auto s = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      cmp_stats[i] = index.search_with_tags(
          query + i * query_aligned_dim, recall_at, L,
          query_result_tags + i * recall_at, nullptr, res);

      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000000;
    }
    for (int64_t i = 0; i < (int64_t) query_num * recall_at; i++) {
      query_result_ids[test_id][i] = *(query_result_tags + i);
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    float qps = (query_num / diff.count());

    float recall = 0;
    if (calc_recall_flag)
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);

    std::sort(latency_stats.begin(), latency_stats.end());
    float mean_latency =
        std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) /
        query_num;

    float avg_cmps =
        (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) /
        (float) query_num;

    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << avg_cmps << std::setw(20) << (float) mean_latency
              << std::setw(15)
              << (float) latency_stats[(_u64) (0.999 * query_num)];
    if (calc_recall_flag)
      std::cout << std::setw(12) << recall;
    std::cout << std::endl;

    // delete[] query_result_tags;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    std::string cur_result_path =
        result_path_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    // diskann::save_bin<_u32>(cur_result_path,
    // query_result_ids[test_id].data(), query_num, recall_at);
    test_id++;
  }
  diskann::aligned_free(query);

  return 0;
}

float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t* res,
                    uint32_t* gt) {
  uint32_t total_count = 0;
  for (uint32_t i = 0; i < q_num; i++) {
    std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
    std::vector<uint32_t> intersection;
    std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
    for (auto p : one_gt) {
      if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end())
        intersection.push_back(p);
    }

    total_count += static_cast<uint32_t>(intersection.size());
  }
  return static_cast<float>(total_count) / (float) (k * q_num);
}

double ComputeRderr(float* gt_dist, uint32_t gt_dim,
                    std::vector<std::vector<float>>& res_dists, uint32_t k,
                    diskann::Metric metric) {
  double   total_err = 0;
  uint32_t q_num = res_dists.size();

  for (uint32_t i = 0; i < q_num; i++) {
    std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
    std::vector<float> temp_res(res_dists[i].begin(), res_dists[i].end());
    if (metric == diskann::INNER_PRODUCT) {
      for (size_t j = 0; j < k; ++j) {
        temp_res[j] = -1.0 * temp_res[j];
      }
    } else if (metric == diskann::COSINE) {
      for (size_t j = 0; j < k; ++j) {
        temp_res[j] = 2.0 * (1.0 - (-1.0 * temp_res[j]));
      }
    }
    double err = 0.0;
    for (uint32_t j = 0; j < k; j++) {
      err += std::fabs(temp_res[j] - one_gt[j]) / double(one_gt[j]);
    }
    err = err / static_cast<double>(k);
    total_err = total_err + err;
  }
  return total_err / static_cast<double>(q_num);
}

int main(int argc, char** argv) {
  std::string base_data_file;
  std::string query_file;
  std::string gt_file;

  std::string           projection_index_save_file;
  std::string           data_type;
  std::string           dist;
  std::vector<uint32_t> L_vec;
  uint32_t              num_threads;
  uint32_t              k;
  std::string           evaluation_save_path = "";

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                       "distance function <l2/ip>");
    desc.add_options()("query_path",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in bin format");
    desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(),
                       "Groundtruth file in bin format");
    desc.add_options()(
        "index_save_path",
        po::value<std::string>(&projection_index_save_file)->required(),
        "Path prefix for saving projetion index file components");
    desc.add_options()(
        "search_list",
        po::value<std::vector<uint32_t>>(&L_vec)->multitoken()->required(),
        "Priority queue length for searching");
    desc.add_options()("k",
                       po::value<uint32_t>(&k)->default_value(1)->required(),
                       "k nearest neighbors");
    desc.add_options()("evaluation_save_path",
                       po::value<std::string>(&evaluation_save_path),
                       "Path prefix for saving evaluation results");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if ((dist == std::string("ip")) && (data_type == std::string("float"))) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                 "supported in general, and mips/fast_l2 only for floating "
                 "point data."
              << std::endl;
    return -1;
  }

  omp_set_num_threads(num_threads);

  float* query_data = nullptr;
  size_t q_pts, q_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<float>(query_file, query_data, q_pts, q_dim,
                                   query_aligned_dim);
  q_dim = query_aligned_dim;

  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
  if (gt_num != q_pts) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  // float* base_data = nullptr;
  // size_t base_num, base_dim, base_aligned_dim;
  // diskann::load_aligned_bin<float>(base_data_file, base_data, base_num,
  //                                  base_dim, base_aligned_dim);

  diskann::Index<float, uint32_t> index(metric, q_dim, 0, true, true, false);
  index.load(projection_index_save_file.c_str(), num_threads,
             *(std::max_element(L_vec.begin(), L_vec.end())));
  std::cout << "Index loaded. Load graph index from: "
            << projection_index_save_file << std::endl;

  // Search
  std::cout << "k: " << k << std::endl;
  uint32_t* res = new uint32_t[q_pts * k];
  float*    res_dists = new float[q_pts * k];
  uint32_t* projection_cmps_vec = new uint32_t[q_pts];
  uint32_t* hops_vec = new uint32_t[q_pts];
  float*    projection_latency_vec = new float[q_pts];
  memset(res, 0, sizeof(uint32_t) * q_pts * k);
  memset(res_dists, 0, sizeof(float) * q_pts * k);
  memset(projection_cmps_vec, 0, sizeof(uint32_t) * q_pts);
  memset(hops_vec, 0, sizeof(uint32_t) * q_pts);
  memset(projection_latency_vec, 0, sizeof(float) * q_pts);
  std::ofstream evaluation_out;
  if (!evaluation_save_path.empty()) {
    evaluation_out.open(evaluation_save_path, std::ios::out);
  }
  std::cout << "Using thread: " << num_threads << std::endl;
  std::cout << "L_pq" << "\t\tQPS" << "\t\t\tavg_visited" << "\tmean_latency"
            << "\trecall@" << k << "\tavg_hops" << std::endl;
  if (evaluation_out.is_open()) {
    evaluation_out << "L_pq,QPS,avg_visited,mean_latency,recall@" << k
                   << ",avg_hops" << std::endl;
  }
  for (uint32_t L_pq : L_vec) {
    if (k > L_pq) {
      std::cout << "L_pq must greater or equal than k" << std::endl;
      exit(1);
    }
    // pre test good
    for (size_t i = 0; i < 100; ++i) {
      index.search(query_data + i * q_dim, k, L_pq, res + i * k,
                   res_dists + i * k);
    }

    // record the search time
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < q_pts; ++i) {
      std::pair<uint32_t, uint32_t> retval = index.search(
          query_data + i * q_dim, k, L_pq, res + i * k, res_dists + i * k);
      hops_vec[i] = retval.first;              // retval.first hops
      projection_cmps_vec[i] = retval.second;  // retval.second cmps
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    float qps = (float) q_pts / ((float) diff / 1000.0);
    float recall = ComputeRecall(q_pts, k, gt_dim, res, gt_ids);
    float avg_projection_cmps = 0.0;
    for (size_t i = 0; i < q_pts; ++i) {
      avg_projection_cmps += projection_cmps_vec[i];
    }
    avg_projection_cmps /= q_pts;

    float avg_hops = 0.0;
    for (size_t i = 0; i < q_pts; ++i) {
      avg_hops += hops_vec[i];
    }
    avg_hops /= (float) q_pts;
    float avg_projection_latency = 0.0;
    for (size_t i = 0; i < q_pts; ++i) {
      avg_projection_latency += projection_latency_vec[i];
    }
    avg_projection_latency /= (float) q_pts;
    std::cout << L_pq << "\t\t" << qps << "\t\t" << avg_projection_cmps
              << "\t\t" << ((float) diff / q_pts) << "\t\t" << recall << "\t\t"
              << avg_hops << std::endl;
    if (evaluation_out.is_open()) {
      evaluation_out << L_pq << "," << qps << "," << avg_projection_cmps << ","
                     << ((float) diff / q_pts) << "," << recall << ","
                     << avg_hops << std::endl;
    }
  }

  if (evaluation_out.is_open()) {
    evaluation_out.close();
  }

  diskann::aligned_free(query_data);
  return 0;
}