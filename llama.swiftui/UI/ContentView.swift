import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var multiLineText = ""
    @State private var showingHelp = false    // To track if Help Sheet should be shown

    var body: some View {
        NavigationView {
            VStack {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }
                
                HStack () {
                    Button("Info") {
                        clear()
                    }
                    
                    Button("Bench") {
                        bench()
                    }
    
                    Button("Clear") {
                        clear()
                    }
                    
                    Button("Copy") {
                        UIPasteboard.general.string = llamaState.messageLog
                    }
                    
                    Button("Stop") {
                        sendText()
                    }
                    
                }
                .buttonStyle(.bordered)
                
                HStack {
                    TextEditor(text: $multiLineText)
                        .frame(height: 42)
                        .border(Color.gray, width: 0.5)
                        .cornerRadius(12)
                        .overlay( /// apply a rounded border
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(.gray, lineWidth: 1)
                        )
                    Button(action: sendText) {
                        Label("", systemImage: "paperplane").imageScale(.large)
                    }
                }.padding()

            }
            .navigationBarTitle("Mokpell mini", displayMode: .inline)
            .toolbar{
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Relise ctx") {
                        relise()
                    }
                    
                }
            }
            .toolbar{
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink(destination: DrawerView(llamaState: llamaState)) {
                        Text("Models")
                    }
                    
                }
            }

        }
    }
    func relise() {
        Task.detached {
            await llamaState.reliseModel()
        }
    }
    
    func stop() {
        Task.detached {
            await llamaState.cancelInference()
        }
    }

    func sendText() {
        Task{
            await llamaState.complete(text: multiLineText)
            multiLineText = ""
        }
    }

    func bench() {
        Task {
            await llamaState.bench()
        }
    }

    func clear() {
        Task {
            await llamaState.clear()
        }
    }
    
    func modelInfo() {
        
    }
    
    struct DrawerView: View {
        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false
        @State private var showingFilePicker = false
        @State private var selectedFileURL: URL? = nil
        
        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = llamaState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                do {
                    try FileManager.default.removeItem(at: fileURL)
                } catch {
                    print("Error deleting file: \(error)")
                }
            }

            // Remove models from downloadedModels array
            llamaState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            return paths[0]
        }
        var body: some View {
            List {
                Section(header: Text("Download Models From Hugging Face")) {
                    HStack {
                        InputButton(llamaState: llamaState)
                    }
                }
                Section(header: Text("Downloaded Models")) {
                    ForEach(llamaState.downloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                    .onDelete(perform: delete)
                }
                Section(header: Text("Default Models")) {
                    ForEach(llamaState.undownloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                }

            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Model Settings", displayMode: .inline)
            .toolbar{
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Load") {
                        showingFilePicker = true // Открываем файловый пикер
                    }
                }
            }.sheet(isPresented: $showingFilePicker) { // Sheet для файлового пикера
                DocumentPicker(isPresented: $showingFilePicker, selectedFileURL: $selectedFileURL, llamaState: llamaState)
            }

        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

// Координатор для обработки выбора файла
class DocumentPickerCoordinator: NSObject, UIDocumentPickerDelegate {
    let parent: DocumentPicker
    
    init(parent: DocumentPicker) {
        self.parent = parent
    }
    
    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        if let url = urls.first {
            guard url.startAccessingSecurityScopedResource() else {
                        print("Failed to access security scoped resource")
                        parent.isPresented = false
                        return
                    }
                    
                    print("Selected file URL: \(url.path)")
                    parent.selectedFileURL = url
                    
                    // Загрузка модели
                    do {
                        try parent.llamaState.loadModelFromFile(modelUrl: url.path)
                        print("Model loaded successfully")
                    } catch {
                        print("Failed to load model: \(error)")
                    }
            
        }
        parent.isPresented = false
    }
    
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        parent.isPresented = false
    }
}

// Представление UIDocumentPickerViewController для SwiftUI
struct DocumentPicker: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    @Binding var selectedFileURL: URL?
    @ObservedObject var llamaState: LlamaState
    
    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        // Создаем пикер для выбора файлов (например, с расширением .gguf)
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.data], asCopy: false)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}
    
    func makeCoordinator() -> DocumentPickerCoordinator {
        return DocumentPickerCoordinator(parent: self)
    }
}
