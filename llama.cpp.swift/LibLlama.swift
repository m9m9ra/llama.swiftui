import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

struct LlamaChatMessage {
    let role: String
    let content: String
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    var is_done: Bool = false

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]

    var n_len: Int32 = 1024
    var n_cur: Int32 = 0
    var maxNewTokens: Int32 = 128
    var n_decode: Int32 = 0

    init(
        model: OpaquePointer,
        context: OpaquePointer,
        n_ctx: UInt32 = 1024,
        n_batch: UInt32 = 512,
        n_threads: Int = 1,
        n_threads_batch: Int = 1,
        max_tokens: Int32 = 128,
        temp: Float = 0.8,
        min_p: Float = 0.05,
        top_p: Float = 0.95,
        top_k: Int32 = 40,
        penalties_last_n: Int32 = 64,
        penalties_repeat: Float = 1.1,
        penalties_freq: Float = 0.1,
        penalties_pres: Float = 0.0
    ) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batch = llama_batch_init(512, 0, 1)
        self.temporary_invalid_cchars = []
        
        // init settings
        self.maxNewTokens = max_tokens;
        
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        
        // init sampler settings
        llama_sampler_chain_add(self.sampling, llama_sampler_init_penalties(penalties_last_n, penalties_repeat, penalties_freq, penalties_pres))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(top_k))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(top_p, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_min_p(min_p, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(temp))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        vocab = llama_model_get_vocab(model)
    }

    deinit {
        is_done = true
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_model_free(model)
        llama_free(context)
        llama_backend_free()
    }
    
    func release_context() {
        is_done = true
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        llama_memory_clear(llama_get_memory(context), true)
        
        // скорее всего это приведет к ошибке
        // при повторном обращении к свободному контексту
        // обязательна реинициализация
        // лучше удалить все ссылки на экземпляр класса
        // в таком случае будет вызван deinit
        // и все ресурсы высвободятся
        
//         llama_sampler_free(sampling)
//         llama_batch_free(batch)
//         llama_model_free(model)
//         llama_free(context)
//         llama_backend_free()
    }

    static func create_context(
        model_path: String,
        n_ctx: UInt32 = 1024,
        n_batch: UInt32 = 512,
        n_threads: Int = 1,
        n_threads_batch: Int = 1,
        max_tokens: Int32 = 128,
        temp: Float = 0.8,
        min_p: Float = 0.05,
        top_p: Float = 0.95,
        top_k: Int32 = 40,
        penalties_last_n: Int32 = 64,
        penalties_repeat: Float = 1.1,
        penalties_freq: Float = 0.1,
        penalties_pres: Float = 0.0,
        use_ggml_backend: Bool = false
    ) throws -> LlamaContext {
        if use_ggml_backend {
            ggml_backend_load_all()
        } else {
            llama_backend_init()
        }
        
        var model_params = llama_model_default_params()

#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        let model = llama_model_load_from_file(model_path, model_params)
        guard let model else {
            print("Could not load model at \(model_path)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        print("Using \(n_threads) threads")

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)

        let context = llama_init_from_model(model, ctx_params)
        
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }

        return LlamaContext(
            model: model,
            context: context,
            n_ctx: n_ctx,
            n_batch: n_batch,
            n_threads: n_threads,
            n_threads_batch: n_threads_batch,
            max_tokens: max_tokens,
            temp: temp,
            min_p: min_p,
            top_p: top_p,
            top_k: top_k,
            penalties_last_n: penalties_last_n,
            penalties_repeat: penalties_repeat,
            penalties_freq: penalties_freq,
            penalties_pres: penalties_pres,
        )
    }
    
    func inference_cancel() {
        print("\n")
        is_done = true
        temporary_invalid_cchars.removeAll()
        n_decode = 0
        print("canceled by user")
    }

    func inference_init(message_list: [LlamaChatMessage], add_bos: Bool = true, clear_token: Bool = true) {
        is_done = false
        print("attempting to complete \"\(message_list)\"")
        let prompt = formatChatPrompt(messages: message_list, addAssistant: true)
        
        tokens_list = tokenize(text: prompt, add_bos: add_bos, clear_token: clear_token)
        temporary_invalid_cchars = []

        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens_list.count + (Int(n_len) - tokens_list.count)

        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")

        if n_kv_req > n_ctx {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }

        for id in tokens_list {
            print(String(cString: token_to_piece(token: id) + [0]))
        }

        llama_batch_clear(&batch)

        for i1 in 0..<tokens_list.count {
            let i = Int(i1)
            llama_batch_add(&batch, tokens_list[i], Int32(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
    }

    func inference_loop() -> String {
        guard !is_done else { return "" }
        var new_token_id: llama_token = 0
        print("curl: \(n_cur)")
        
        new_token_id = llama_sampler_sample(sampling, context, batch.n_tokens - 1)
        
        llama_sampler_accept(sampling, new_token_id)
        
        let n_ctx = Int(llama_n_ctx(context))
            let n_ctx_used = tokens_list.count
            guard n_ctx_used + 1 <= n_ctx else {
            print("Context size exceeded: \(n_ctx_used) + 1 > \(n_ctx)")
            is_done = true
            return ""
        }

        if llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len || n_decode >= maxNewTokens {
            print("\n")
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            n_decode = 0
            return new_token_str
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        
        print("Sampled token ID: \(new_token_id)")
        print(new_token_str)
        
        tokens_list.append(new_token_id)

        llama_batch_clear(&batch)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
        }

        return new_token_str
    }

    func clear() {
        is_done = true
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        llama_memory_clear(llama_get_memory(context), true)
    }
    
    
    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }
    
    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            // bench prompt processing

            llama_batch_clear(&batch)

            let n_tokens = pp

            for i in 0..<n_tokens {
                llama_batch_add(&batch, 0, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1 // true

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000;

            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }
            llama_synchronize(context)

            let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000;

            // bench text generation

            llama_memory_clear(llama_get_memory(context), false)

            let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000;

            for i in 0..<tg {
                llama_batch_clear(&batch)

                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true)
                }

                if llama_decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
                llama_synchronize(context)
            }

            let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000;

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = model_info();
        let model_size     = String(format: "%.2f GiB", Double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0);
        let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(model)) / 1e9);
        let backend        = "Metal";
        let pp_avg_str     = String(format: "%.2f", pp_avg);
        let tg_avg_str     = String(format: "%.2f", tg_avg);
        let pp_std_str     = String(format: "%.2f", pp_std);
        let tg_std_str     = String(format: "%.2f", tg_std);

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result;
    }

    private func tokenize(text: String, add_bos: Bool, clear_token: Bool = true) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, clear_token)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
    
    func formatChatPrompt(messages: [LlamaChatMessage], addAssistant: Bool = true) -> String {
        let tmpl = llama_model_chat_template(model, nil) // Используем шаблон модели Qwen
        let n_msg = messages.count
        
        // Создаем массив структур llama_chat_message
        var chat: [llama_chat_message] = []
        
        // Храним C-строки для предотвращения их освобождения до использования
        var cStrings: [(role: [UnsafePointer<CChar>], content: [UnsafePointer<CChar>])] = []
        cStrings.reserveCapacity(n_msg)
        
        chat = messages.map { llama_chat_message(role: strdup($0.role), content: strdup($0.content)) }
        
        print(chat)
        let bufSize = 2 * messages.reduce(0) { $0 + $1.content.utf8.count + $1.role.utf8.count } + 1024
        let buf = UnsafeMutablePointer<CChar>.allocate(capacity: bufSize)
        defer { buf.deallocate() }
        let newLen = llama_chat_apply_template(tmpl, chat, n_msg, addAssistant, buf, Int32(bufSize))
        guard newLen >= 0 else {
            print("Failed to apply chat template")
            return ""
        }
        
        return String(cString: buf, encoding: .utf8) ?? ""
    }
}
