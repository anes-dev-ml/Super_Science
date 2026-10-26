#include "tokenizer/qwen3_tokenizer.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
int main(){
    // A controlled vocabulary with every ByteLevel byte; verifies round-trips,
    // special token IDs, UTF-8, and the chat envelope without model weights.
    const auto p=std::filesystem::temp_directory_path()/"super_science_native_tokenizer_fixture.json";
    nlohmann::json j; j["model"]["type"]="BPE";j["model"]["vocab"]=nlohmann::json::object();j["model"]["merges"]=nlohmann::json::array();
    // Original GPT-2 ByteLevel byte-to-Unicode mapping.
    bool taken[256]={};for(int b=33;b<=126;++b)taken[b]=true;for(int b=161;b<=172;++b)taken[b]=true;for(int b=174;b<=255;++b)taken[b]=true;
    auto utf8=[](int x){std::string s;if(x<128)s+=char(x);else if(x<2048){s+=char(0xc0|(x>>6));s+=char(0x80|(x&63));}else{s+=char(0xe0|(x>>12));s+=char(0x80|((x>>6)&63));s+=char(0x80|(x&63));}return s;};
    int extra=0;for(int b=0;b<256;++b){int cp=taken[b]?b:256+extra++;j["model"]["vocab"][utf8(cp)]=b;}
    j["added_tokens"]=nlohmann::json::array({{{"id",256},{"content","<|im_start|>"},{"special",true}},{{"id",257},{"content","<|im_end|>"},{"special",true}},{{"id",151667},{"content","<think>"},{"special",false}},{{"id",151668},{"content","</think>"},{"special",false}}});
    {std::ofstream f(p);f<<j.dump();}
    Qwen3Tokenizer t(p.string());
    for(const auto& s:{std::string("Hello, world!"),std::string("it's WE'RE   123\n OK"),std::string("François 日本語 مرحبا 😀"),std::string(" a\t  \r\n")}){
        auto tokens=t.encode(s);assert(!tokens.empty());assert(t.decode(tokens)==s);
    }
    auto chat=t.chatPrompt("Hello");assert(chat.front()==256);assert(t.special("<|im_end|>")==257);
    assert(t.decode({151667, 151668}, false) == "<think></think>");
    assert(t.decode({151667, 151668}, true).empty());
    const auto expected = std::string("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert(t.decode(chat, false) == expected);
    const std::vector<std::pair<std::string, std::string>> turns{
        {"user", "Hello"}, {"assistant", "Hi!"}, {"user", "Again?"}};
    const auto multi = t.chatConversation(turns);
    assert(t.decode(multi, false) ==
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi!<|im_end|>\n"
        "<|im_start|>user\nAgain?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    bool invalidRoles = false;
    try { t.chatConversation({{"assistant", "not a user turn"}}); }
    catch (const std::invalid_argument &) { invalidRoles = true; }
    assert(invalidRoles);
    const auto thinkingChat = t.chatPrompt("Hello", true);
    assert(t.decode(thinkingChat, false) == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    bool rejected=false;try{t.encode(std::string("\xc0\x80",2));}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    std::filesystem::remove(p);
    std::cout<<"Native tokenizer round-trip and chat tests PASSED\n";
}
