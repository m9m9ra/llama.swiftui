import Flutter
import UIKit
import llama // Импортируй библиотеку llama
import Metal
import CommonCrypto

public class LibLlamaPlugin: NSObject, FlutterPlugin, FlutterStreamHandler {
    private var llamaContexts: [NSNumber: LlamaContext] = [:]
    private var eventSink: FlutterEventSink?
    private let llamaQueue = DispatchQueue(label: "lib.llama.swift", attributes: .serial)

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "lib.llama.swift", binaryMessenger: registrar.messenger())
        let eventChannel = FlutterEventChannel(name: "lib.llama.swift.event", binaryMessenger: registrar.messenger())
        let instance = MyLlamaPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
        eventChannel.setStreamHandler(instance)
    }

    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        eventSink = nil
        return nil
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initContext":
            handleInitContext(call: call, result: result)
        case "completion":
            handleCompletion(call: call, result: result)
        case "stopCompletion":
            handleStopCompletion(call: call, result: result)
        case "tokenize":
            handleTokenize(call: call, result: result)
        case "detokenize":
            handleDetokenize(call: call, result: result)
        case "bench":
            handleBench(call: call, result: result)
        case "releaseContext":
            handleReleaseContext(call: call, result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func handleInitContext(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let params = call.arguments as? [String: Any],
              let modelPath = params["modelPath"] as? String else {
            result(FlutterError(code: "INVALID_ARGS", message: "Model path is required", details: nil))
            return
        }

        let emitLoadProgress = params["emit_load_progress"] as? Bool ?? false
        let useGGMLBackend = params["use_ggml_backend"] as? Bool ?? false
        let nCtx = params["n_ctx"] as? UInt32 ?? 1024
        let nBatch = params["n_batch"] as? UInt32 ?? 512
        let maxTokens = params["max_tokens"] as? Int32 ?? 128
        let temp = params["temp"] as? Float ?? 0.8
        let minP = params["min_p"] as? Float ?? 0.05
        let topP = params["top_p"] as? Float ?? 0.95
        let topK = params["top_k"] as? Int32 ?? 40
        let penaltiesLastN = params["penalties_last_n"] as? Int32 ?? 64
        let penaltiesRepeat = params["penalties_repeat"] as? Float ?? 1.1
        let penaltiesFreq = params["penalties_freq"] as? Float ?? 0.1
        let penaltiesPres = params["penalties_pres"] as? Float ?? 0.0

        do {
            let context = try LlamaContext.create_context(
                model_path: modelPath,
                n_ctx: nCtx,
                n_batch: nBatch,
                max_tokens: maxTokens,
                temp: temp,
                min_p: minP,
                top_p: topP,
                top_k: topK,
                penalties_last_n: penaltiesLastN,
                penalties_repeat: penaltiesRepeat,
                penalties_freq: penaltiesFreq,
                penalties_pres: penaltiesPres,
                use_ggml_backend: useGGMLBackend
            )

            // Проверяем поддержку Metal
            let metalDevice = MTLCreateSystemDefaultDevice()
            let isMetalEnabled = metalDevice != nil
            let reasonNoMetal = isMetalEnabled ? "" : "Metal not supported on this device"

            // Генерируем уникальный contextId
            let contextId = Double.random(in: 0..<1000000)
            let contextIdNumber = NSNumber(value: contextId)
            llamaContexts[contextIdNumber] = context

            // Отправляем прогресс загрузки, если требуется
            if emitLoadProgress {
                DispatchQueue.main.async { [weak self] in
                    self?.eventSink?([
                        "function": "loadProgress",
                        "contextId": "",
                        "result": 100 // Предполагаем, что загрузка завершена
                    ])
                }
            }

            result([
                "contextId": contextIdNumber,
                "gpu": isMetalEnabled,
                "reasonNoGPU": reasonNoMetal,
                "model": context.model_info()
            ])
        } catch {
            result(FlutterError(code: "INIT_ERROR", message: "Failed to initialize Llama: \(error)", details: nil))
        }
    }

    private func handleCompletion(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let contextId = args["contextId"] as? Double,
              let params = args["params"] as? [String: Any],
              let messages = params["messages"] as? [[String: String]] else {
            result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let context = llamaContexts[contextIdNumber] else {
            result(FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        let emitRealtimeCompletion = params["emit_realtime_completion"] as? Bool ?? false

        llamaQueue.async { [weak self] in
            let chatMessages = messages.map { LlamaChatMessage(role: $0["role"] ?? "", content: $0["content"] ?? "") }
            context.inference_init(message_list: chatMessages)

            var completionResult = ""
            while !context.is_done {
                let token = context.inference_loop()
                completionResult += token

                if emitRealtimeCompletion {
                    DispatchQueue.main.async {
                        self?.eventSink?([
                            "function": "completion",
                            "contextId": contextId,
                            "result": ["text": token]
                        ])
                    }
                }
            }

            result(["text": completionResult])
        }
    }

    private func handleStopCompletion(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let contextId = args["contextId"] as? Double else {
            result(FlutterError(code: "INVALID_ARGS", message: "Context ID is required", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let context = llamaContexts[contextIdNumber] else {
            result(FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        context.inference_cancel()
        result(nil)
    }

    private func handleTokenize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let contextId = args["contextId"] as? Double,
              let text = args["text"] as? String else {
            result(FlutterError(code: "INVALID_ARGS", message: "Context ID and text are required", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let context = llamaContexts[contextIdNumber] else {
            result(FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        let tokens = context.tokenize(text: text, add_bos: true)
        result(["tokens": tokens.map { Int($0) }])
    }

    private func handleDetokenize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let contextId = args["contextId"] as? Double,
              let tokens = args["tokens"] as? [Int] else {
            result(FlutterError(code: "INVALID_ARGS", message: "Context ID and tokens are required", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let context = llamaContexts[contextIdNumber] else {
            result(FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        let detokenized = tokens.map { context.token_to_piece(token: llama_token($0)) }
            .flatMap { $0 }
            .map { Character(UnicodeScalar(UInt8($0))) }
            .reduce("") { $0 + String($1) }
        result(detokenized)
    }

    private func handleBench(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let contextId = args["contextId"] as? Double,
              let pp = args["pp"] as? Double,
              let tg = args["tg"] as? Double,
              let pl = args["pl"] as? Double,
              let nr = args["nr"] as? Double else {
            result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let context = llamaContexts[contextIdNumber] else {
            result(FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        let benchResult = context.bench(pp: Int(pp), tg: Int(tg), pl: Int(pl), nr: Int(nr))
        result(benchResult)
    }

    private func handleReleaseContext(call_id: String, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let contextId = args["contextId"] as? Double else {
            result(FlutterError(code: "INVALID_ARGS", message: "Context ID is required", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let context = llamaContexts[contextIdNumber] else {
            result(FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        context.inference_cancel()
        llamaQueue.sync {}
        context.release_context()
        llamaContexts.removeValue(forKey: contextIdNumber)
        result(nil)
    }
}
