import 'package:flutter/services.dart';

class LibLlama {
  LibLlama._();
  static LibLlama instance = LibLlama._();
  factory LibLlama () => instance;

  static const MethodChannel channel = MethodChannel('lib.llama');
  static const EventChannel eventChannel = EventChannel('lib.llama.event');
  static bool initialized = false;
  

  Future<Map<String, dynamic>> initializeContext({
    required String modelPath,
    bool emitLoadProgress = false,
    bool useGGMLBackend = false,
    int nTokens = 1024,
    int nBatchSize = 512,
    int? maxTokens,
    double? temperature = 0.8,
    double? minP = 0.05,
    double? topP = 0.95,
    int? topK = 40,
    int? penaltiesLastN = 64,
    double? penaltiesRepeat = 1.1,
    double? penaltiesFreq = 0.1,
    double? penaltiesPres = 0.0,
  }) async {
    if (initialized) {
      throw Exception('Failed to initialize context: context always created');
    }
    try {
      final result = await channel.invokeMapMethod<String, dynamic>('initContext', {
        'modelPath': modelPath,
        'emit_load_progress': emitLoadProgress,
        'use_ggml_backend': useGGMLBackend,
        'n_ctx': nTokens,
        'n_batch': nBatchSize,
        'max_tokens': maxTokens,
        'temp': temperature,
        'min_p': minP,
        'top_p': topP,
        'top_k': topK,
        'penalties_last_n': penaltiesLastN,
        'penalties_repeat': penaltiesRepeat,
        'penalties_freq': penaltiesFreq,
        'penalties_pres': penaltiesPres,
      });
      return result!;
    } catch (e) {
      throw Exception('Failed to initialize context: $e');
    }
  }

  Future<Map<String, dynamic>> completion({
    required String contextId,
    required List<Map<String, String>> messages,
    bool emitRealtimeCompletion = true,
  }) async {
    try {
      final result = await channel.invokeMapMethod<String, dynamic>('completion', {
        'contextId': double.parse(contextId),
        'params': {
          'messages': messages,
          'emit_realtime_completion': emitRealtimeCompletion,
        },
      });
      return result!;
    } catch (e) {
      throw Exception('Failed to run completion: $e');
    }
  }

  Future<void> stopCompletion(String contextId) async {
    try {
      await channel.invokeMethod('stopCompletion', {'contextId': double.parse(contextId)});
    } catch (e) {
      throw Exception('Failed to stop completion: $e');
    }
  }

  Future<Map<String, dynamic>> tokenize({
    required String contextId,
    required String text,
  }) async {
    try {
      final result = await channel.invokeMapMethod('tokenize', {
        'contextId': double.parse(contextId),
        'text': text,
      });
      return result as Map<String, dynamic>;
    } catch (e) {
      throw Exception('Failed to tokenize: $e');
    }
  }

  Future<String> detokenize({
    required String contextId,
    required List<String> tokens,
  }) async {
    try {
      final result = await channel.invokeMethod('detokenize', {
        'contextId': double.parse(contextId),
        'tokens': tokens.map(int.parse).toList(),
      });
      return result as String;
    } catch (e) {
      throw Exception('Failed to detokenize: $e');
    }
  }

  Future<String> bench({
    required String contextId,
    required int pp,
    required int tg,
    required int pl,
    required int nr,
  }) async {
    try {
      final result = await channel.invokeMethod('bench', {
        'contextId': double.parse(contextId),
        'pp': pp,
        'tg': tg,
        'pl': pl,
        'nr': nr,
      });
      return result as String;
    } catch (e) {
      throw Exception('Failed to run bench: $e');
    }
  }

  Future<void> releaseContext() async {
    try {
      await channel.invokeMethod('releaseContext', {'contextId': ''});
    } catch (e) {
      throw Exception('Failed to release context: $e');
    }
  }

  Stream get eventStream {
    return eventChannel.receiveBroadcastStream();
  }
}