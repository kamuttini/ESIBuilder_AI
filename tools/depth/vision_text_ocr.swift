import AppKit
import Foundation
import Vision

guard CommandLine.arguments.count == 6 else {
    fputs("usage: vision_text_ocr IMAGE LEFT TOP RIGHT BOTTOM\n", stderr)
    exit(2)
}

let imagePath = CommandLine.arguments[1]
let values = CommandLine.arguments[2...5].compactMap { Double($0) }
guard values.count == 4,
      let image = NSImage(contentsOfFile: imagePath),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    exit(3)
}

let imageWidth = cgImage.width
let imageHeight = cgImage.height
let left = max(0, min(imageWidth - 1, Int(values[0].rounded())))
let top = max(0, min(imageHeight - 1, Int(values[1].rounded())))
let right = max(left + 1, min(imageWidth, Int(values[2].rounded())))
let bottom = max(top + 1, min(imageHeight, Int(values[3].rounded())))
let cropRect = CGRect(x: left, y: top, width: right - left, height: bottom - top)
guard let cropped = cgImage.cropping(to: cropRect) else {
    exit(4)
}

let request = VNRecognizeTextRequest { request, _error in
    for observation in request.results as? [VNRecognizedTextObservation] ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let box = observation.boundingBox
        let x = Double(left) + Double(box.origin.x) * Double(cropped.width)
        let y = Double(top) + (1.0 - Double(box.origin.y + box.height)) * Double(cropped.height)
        let width = Double(box.width) * Double(cropped.width)
        let height = Double(box.height) * Double(cropped.height)
        let text = candidate.string.replacingOccurrences(of: "\t", with: " ").replacingOccurrences(of: "\n", with: " ")
        print("\(text)\t\(candidate.confidence)\t\(x)\t\(y)\t\(width)\t\(height)")
    }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.recognitionLanguages = ["en_US"]
request.customWords = ["cm", "mm", "Depth"]
request.minimumTextHeight = 0.004

let handler = VNImageRequestHandler(cgImage: cropped, orientation: .up, options: [:])
try handler.perform([request])
