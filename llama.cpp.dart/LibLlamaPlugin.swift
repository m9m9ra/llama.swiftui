import CommonCrypto
import Flutter
import Metal
import UIKit
import llama

@available(iOS 13, *) // if #available(iOS 13, *)
public class LibLlamaPlugin: NSObject, FlutterStreamHandler, ObservableObject {
    private let channelName = "lib.llama"
    private let eventChannelName = "lib.llama.event"

    private var llamaContext: LlamaContext?
    
    private var eventSink: FlutterEventSink?
    private let llamaQueue = DispatchQueue(label: "lib.llama")

    private let methodChannel: FlutterMethodChannel
    private let eventChannel: FlutterEventChannel

    private var messageList: [LlamaChatMessage] = []

    init(messenger: FlutterBinaryMessenger, eventChannel: FlutterEventChannel) {
        let taskQueue = messenger.makeBackgroundTaskQueue?()

        methodChannel = FlutterMethodChannel(
            name: "lib.llama",
            binaryMessenger: messenger,
            codec: FlutterStandardMethodCodec.sharedInstance(),
            taskQueue: taskQueue
        )
        self.eventChannel = eventChannel
    }

    func setMethodCallHandler() {
        methodChannel.setMethodCallHandler(handle)
    }

    private func handle(call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initContext":
            self.handleInitContext(call: call, result: result)
        case "completion":
            self.handleCompletion(call: call, result: result)
        case "stopCompletion":
            self.handleStopCompletion(call: call, result: result)
        case "tokenize":
            self.handleTokenize(call: call, result: result)
        case "detokenize":
            self.handleDetokenize(call: call, result: result)
        case "bench":
            self.handleBench(call: call, result: result)
        case "releaseContext":
            self.handleReleaseContext(call: call, result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    public func onListen(
        withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink
    ) -> FlutterError? {
        self.eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        return nil
    }

    public func handleInitContext(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let params = call.arguments as? [String: Any],
            let modelPath = params["modelPath"] as? String
        else {
            result(
                FlutterError(code: "INVALID_ARGS", message: "Model path is required", details: nil))
            return
        }
        // Если ты долбоеб попробуешь дважды заинитить контекст я тебя защищу ^_^
        llamaContext = nil

        // Но ты все же странный если читаешь че я тут наговнакодил
        let useGGMLBackend = params["use_ggml_backend"] as? Bool ?? false
        let nCtx = params["n_ctx"] as? UInt32 ?? 1024
        
        // а еще если ты воткнешь сюда 64 - то это не будет работать никогда
        let nBatch = params["n_batch"] as? UInt32 ?? 512
        let maxTokens = params["max_tokens"] as? Int32 ?? 256
        let temp = params["temp"] as? Float ?? 0.8
        let minP = params["min_p"] as? Float ?? 0.05
        let topP = params["top_p"] as? Float ?? 0.95
        let topK = params["top_k"] as? Int32 ?? 40
        let penaltiesLastN = params["penalties_last_n"] as? Int32 ?? 64
        let penaltiesRepeat = params["penalties_repeat"] as? Float ?? 1.1
        let penaltiesFreq = params["penalties_freq"] as? Float ?? 0.1
        let penaltiesPres = params["penalties_pres"] as? Float ?? 0.0

        do {
            llamaContext = try LlamaContext.create_context(
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

            let metalDevice = MTLCreateSystemDefaultDevice()
            let isMetalEnabled = metalDevice != nil
            let reasonNoMetal = isMetalEnabled ? "" : "Metal not supported on this device"

            result([
                "gpu": isMetalEnabled,
                "reasonNoGPU": reasonNoMetal,
                "model": "model_info",
            ])
        } catch {
            result(
                FlutterError(
                    code: "INIT_ERROR", message: "Failed to initialize Llama: \(error)",
                    details: nil))
        }
    }

    public func handleCompletion(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
            let params = args["params"] as? [String: Any],
            let messages = params["messages"] as? [[String: String]]
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
            return
        }
        

        let emitRealtimeCompletion = params["emit_realtime_completion"] as? Bool ?? true
        guard let llamaContext else {
            return
        }
        

            Task.detached {
                // Ну если ты все же попытаешься дважды запустить инфиренс - то ты точно странный тип
                if await !llamaContext.is_done {
                    result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
                    return
                }
                
                // а вот стрингой ли они приходят - это уже вопрос хороший
                let chatMessages = messages.map {
                    LlamaChatMessage(role: String($0["role"] ?? "") , content: String($0["message"] ?? ""))
                }
                await llamaContext.inference_init(message_list: chatMessages)

                var completionResult = ""

                // Основной цикл генерации
                while await !llamaContext.is_done {
                    let token = await llamaContext.inference_loop()
                    completionResult += token

                    if emitRealtimeCompletion {
                        DispatchQueue.main.async { [weak self] in
                            self!.eventSink?([
                                "done": false,
                                "function": "completion",
                                "result": ["text": completionResult],
                            ])
                        }
                    }
                }

                DispatchQueue.main.async {
                    result([
                        "done": true,
                        "function": "completion",
                        "result": ["text": completionResult],
                    ])
                }
                await llamaContext.clear()
            }
    }

    private func handleStopCompletion(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
            let contextId = args["contextId"] as? Double
        else {
            result(
                FlutterError(code: "INVALID_ARGS", message: "Context ID is required", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let llamaContext else {
            result(
                FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }
        Task {
            await llamaContext.inference_cancel()
            DispatchQueue.main.async {
                result(nil)
            }
        }

        result(nil)
    }

    private func handleReleaseContext(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any]
        else {
            result(
                FlutterError(code: "INVALID_ARGS", message: "Context ID is required", details: nil))
            return
        }

        llamaContext = nil
        result(nil)
    }

    private func handleTokenize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
            let contextId = args["contextId"] as? Double,
            let text = args["text"] as? String
        else {
            result(
                FlutterError(
                    code: "INVALID_ARGS", message: "Context ID and text are required", details: nil)
            )
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let llamaContext else {
            result(
                FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }

        Task {
            let tokens = await llamaContext.tokenize(text: text, add_bos: true)
            DispatchQueue.main.async {
                result(["tokens": tokens.map { Int($0) }])
            }
        }
    }

    private func handleDetokenize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
            let contextId = args["contextId"] as? Double,
            let tokens = args["tokens"] as? [Int]
        else {
            result(
                FlutterError(
                    code: "INVALID_ARGS", message: "Context ID and tokens are required",
                    details: nil))
            return
        }
    }

    private func handleBench(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
            let contextId = args["contextId"] as? Double,
            let pp = args["pp"] as? Double,
            let tg = args["tg"] as? Double,
            let pl = args["pl"] as? Double,
            let nr = args["nr"] as? Double
        else {
            result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
            return
        }

        let contextIdNumber = NSNumber(value: contextId)
        guard let llamaContext else {
            result(
                FlutterError(code: "CONTEXT_NOT_FOUND", message: "Context not found", details: nil))
            return
        }
        Task {
            let benchResult = await llamaContext.bench(
                pp: Int(pp), tg: Int(tg), pl: Int(pl), nr: Int(nr))
            DispatchQueue.main.async {
                result("benchResult")
            }
        }

    }
}
