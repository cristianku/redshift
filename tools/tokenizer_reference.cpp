// Vocabulary-only oracle: no context, model weights, or GPU inference.
#include "llama.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.load_mtp = false;
    auto *model = llama_model_load_from_file(argv[1], params);
    if (!model) return 1;
    const auto *vocab = llama_model_get_vocab(model);
    for (int i = 2; i < argc; ++i) {
        std::ifstream input(argv[i], std::ios::binary);
        if (!input) { llama_model_free(model); return 1; }
        std::string text((std::istreambuf_iterator<char>(input)), {});
        int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
        std::vector<llama_token> ids(n < 0 ? -n : n);
        n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, true);
        if (n < 0) { llama_model_free(model); return 1; }
        std::cout << '[';
        for (int j = 0; j < n; ++j) std::cout << (j ? "," : "") << ids[j];
        std::cout << "]\n";
    }
    llama_model_free(model);
}
