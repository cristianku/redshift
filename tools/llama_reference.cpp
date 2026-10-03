// Standalone reference harness; links the server's existing llama.cpp libraries.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
    try {
        require(argc==5,"usage: llama-reference MODEL PREFIX REPEATS OUTPUT_STEM");
        int prefix=std::stoi(argv[2]), repeats=std::stoi(argv[3]);
        require(prefix>=0 && prefix<=2040 && repeats>0,"invalid benchmark dimensions");
        const std::string stem=argv[4];
        ggml_backend_load_all();
        llama_backend_init();
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=-1;
        mp.load_mtp=false;
        mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        auto start=Clock::now();
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1],mp),llama_model_free);
        require(bool(model),"model load failed");
        int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(vocab==248320,"unexpected vocabulary");
        auto cp=llama_context_default_params();
        cp.n_ctx=prefix+8;
        cp.n_batch=8; cp.n_ubatch=8; cp.n_seq_max=1;
        cp.n_outputs_max=8; cp.n_outputs_max_per_seq=8;
        cp.n_threads=20; cp.n_threads_batch=20;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
        cp.offload_kqv=true; cp.op_offload=true;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(),cp),llama_free);
        require(bool(ctx),"context creation failed");
        double load_seconds=seconds(start,Clock::now());
        auto evaluate=[&](const std::vector<int> &ids,int position) {
            llama_batch batch=llama_batch_init(int(ids.size()),0,1);
            batch.n_tokens=int(ids.size());
            for(int t=0;t<batch.n_tokens;t++) {
                batch.token[t]=ids[t]; batch.pos[t]=position+t;
                batch.n_seq_id[t]=1; batch.seq_id[t][0]=0; batch.logits[t]=1;
            }
            int rc=llama_decode(ctx.get(),batch);
            llama_batch_free(batch);
            require(rc==0,"llama_decode failed");
            llama_synchronize(ctx.get());
            std::vector<float> out(ids.size()*vocab);
            for(size_t t=0;t<ids.size();t++) {
                const float *row=llama_get_logits_ith(ctx.get(),int(t));
                require(row!=nullptr,"missing logit row");
                std::memcpy(out.data()+t*vocab,row,size_t(vocab)*sizeof(float));
            }
            return out;
        };
        for(int p=0;p<prefix;p+=8) {
            std::vector<int> ids;
            for(int i=p;i<std::min(prefix,p+8);i++) ids.push_back(10+i);
            evaluate(ids,p);
        }
        const auto flags=LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
        size_t size=llama_state_seq_get_size_ext(ctx.get(),0,flags);
        require(size>0,"checkpoint sizing failed");
        std::vector<uint8_t> checkpoint(size);
        require(llama_state_seq_get_data_ext(ctx.get(),checkpoint.data(),size,0,flags)==size,
                "checkpoint save failed");
        llama_synchronize(ctx.get());
        auto restore=[&] {
            require(llama_state_seq_set_data_ext(ctx.get(),checkpoint.data(),size,0,flags)==size,
                    "checkpoint restore failed");
            llama_synchronize(ctx.get());
        };
        const std::vector<int> tokens={100,113,126,139,152,165,178,191};
        restore();
        std::vector<float> sequential;
        for(int t=0;t<8;t++) {
            auto row=evaluate({tokens[t]},prefix+t);
            sequential.insert(sequential.end(),row.begin(),row.end());
        }
        std::ofstream logits(stem+".f32",std::ios::binary);
        logits.write(reinterpret_cast<const char *>(sequential.data()),
                     std::streamsize(sequential.size()*sizeof(float)));
        require(bool(logits),"cannot write reference logits");
        logits.close();
        double max_group_error=0;
        int group_argmax_mismatches=0;
        for(int n : {1,2,4,8}) {
            restore();
            auto rows=evaluate(std::vector<int>(tokens.begin(),tokens.begin()+n),prefix);
            for(size_t i=0;i<rows.size();i++) {
                require(std::isfinite(rows[i]) && std::isfinite(sequential[i]),"non-finite logits");
                max_group_error=std::max(max_group_error,double(std::abs(rows[i]-sequential[i]))/(1+std::abs(sequential[i])));
            }
            for(int t=0;t<n;t++) {
                auto a=rows.begin()+t*vocab, b=sequential.begin()+t*vocab;
                group_argmax_mismatches+=(std::max_element(a,a+vocab)-a != std::max_element(b,b+vocab)-b);
            }
        }
        std::ofstream report(stem+".json");
        require(bool(report),"cannot write report");
        report.precision(17);
        report << "{\"engine\":\"llama.cpp\",\"prefix\":" << prefix
               << ",\"actual_context\":" << llama_n_ctx(ctx.get())
               << ",\"load_seconds\":" << load_seconds
               << ",\"checkpoint_host_metadata_bytes\":" << size
               << ",\"checkpoint_storage\":\"GPU\",\"flash_attention\":true,\"kv_type\":\"f16\""
               << ",\"all_logit_rows\":true,\"max_group_scaled_error\":" << max_group_error
               << ",\"group_argmax_mismatches\":" << group_argmax_mismatches << ",\"samples\":[";
        bool first=true;
        double output_sum=0;
        for(int repeat=0;repeat<repeats;repeat++) {
            std::vector<int> order={1,2,4,8};
            if(repeat%2) std::reverse(order.begin(),order.end());
            for(int n:order) {
                auto begin=Clock::now(); restore(); auto restored=Clock::now();
                auto rows=evaluate(std::vector<int>(tokens.begin(),tokens.begin()+n),prefix);
                auto end=Clock::now();
                output_sum+=rows[0];
                if(!first) report << ',';
                first=false;
                report << "{\"repeat\":" << repeat << ",\"batch\":" << n
                       << ",\"restore_seconds\":" << seconds(begin,restored)
                       << ",\"evaluate_seconds\":" << seconds(restored,end)
                       << ",\"total_seconds\":" << seconds(begin,end) << '}';
            }
        }
        report << "],\"output_guard\":" << output_sum << "}\n";
        require(bool(report),"report write failed");
        return 0;
    } catch(const std::exception &error) {
        std::fprintf(stderr,"reference error: %s\n",error.what());
        return 1;
    }
}
