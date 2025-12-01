import { useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import {
  Upload,
  CloudUpload,
  FileImage,
  X,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  Zap,
  Server,
  Globe,
  Eye,
  Timer,
  Database,
  Container,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

// ============================================================================
// API CONFIGURATION - Update these URLs as needed
// ============================================================================
const UPLOAD_URL =
  "https://4uck7vb3bd.execute-api.us-east-1.amazonaws.com/prod/upload-image";
const RESULTS_BASE_URL =
  "https://22xfrjkjw2.execute-api.us-east-1.amazonaws.com/dev/results";

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/** Response from the upload-image endpoint */
interface UploadResponse {
  request_id: string;
  bucket: string;
  s3_key: string;
  content_type: string;
}

/** Response from the results endpoint */
interface InferenceResult {
  id: number;
  image_name: string;
  predicted_label: string;
  confidence: number;
  created_at: string;
  request_id: string;
  infer_latency_ms: number;
}

/** Status of each image being processed */
type ImageStatus = "pending" | "uploading" | "polling" | "completed" | "error" | "timeout";

/** State for each image in the queue */
interface ImageState {
  file: File;
  status: ImageStatus;
  requestId?: string;
  result?: InferenceResult;
  error?: string;
  previewUrl?: string; // Data URL for image preview
  startTime?: number; // Timestamp when processing started (for E2E latency)
  e2eLatencyMs?: number; // End-to-end latency in milliseconds
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/** Format file size to human readable string */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Convert File to data URL for preview */
function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ============================================================================
// MAIN APPLICATION COMPONENT
// ============================================================================

export default function App() {
  // State for selected files before upload
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  // State for images being processed (upload + inference)
  const [imageStates, setImageStates] = useState<Map<string, ImageState>>(
    new Map()
  );
  // Whether upload is in progress
  const [isUploading, setIsUploading] = useState(false);
  // Drag and drop state
  const [isDragActive, setIsDragActive] = useState(false);
  // Image preview dialog state
  const [previewImage, setPreviewImage] = useState<{
    url: string;
    name: string;
    prediction?: string;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // --------------------------------------------------------------------------
  // FILE SELECTION HANDLERS
  // --------------------------------------------------------------------------

  const handleFileSelect = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (files) {
        const fileArray = Array.from(files).filter((f) =>
          f.type.startsWith("image/")
        );
        setSelectedFiles((prev) => [...prev, ...fileArray]);
      }
    },
    []
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files) {
      const fileArray = Array.from(files).filter((f) =>
        f.type.startsWith("image/")
      );
      if (fileArray.length === 0) {
        toast.error("Please drop image files only");
        return;
      }
      setSelectedFiles((prev) => [...prev, ...fileArray]);
    }
  }, []);

  const removeFile = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const clearAllFiles = useCallback(() => {
    setSelectedFiles([]);
  }, []);

  // --------------------------------------------------------------------------
  // UPLOAD AND POLLING LOGIC
  // --------------------------------------------------------------------------

  /**
   * Upload a single image and start polling for results
   */
  const processImage = async (file: File, fileKey: string, previewUrl: string) => {
    // Record start time for E2E latency calculation
    const startTime = Date.now();

    // Update state to uploading with preview URL and start time
    setImageStates((prev) => {
      const next = new Map(prev);
      next.set(fileKey, { file, status: "uploading", previewUrl, startTime });
      return next;
    });

    try {
      // Step 1: Upload the image
      const uploadRes = await fetch(UPLOAD_URL, {
        method: "POST",
        headers: {
          "Content-Type": file.type || "image/jpeg",
        },
        body: file,
      });

      if (!uploadRes.ok) {
        throw new Error(`Upload failed: ${uploadRes.statusText}`);
      }

      const uploadData: UploadResponse = await uploadRes.json();
      const requestId = uploadData.request_id;

      // Update state to polling
      setImageStates((prev) => {
        const next = new Map(prev);
        next.set(fileKey, { file, status: "polling", requestId, previewUrl, startTime });
        return next;
      });

      // Step 2: Poll for results
      await pollForResult(fileKey, requestId, file, previewUrl, startTime);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      setImageStates((prev) => {
        const next = new Map(prev);
        const existing = prev.get(fileKey);
        next.set(fileKey, { 
          file, 
          status: "error", 
          error: errorMessage,
          previewUrl: existing?.previewUrl,
          startTime: existing?.startTime,
        });
        return next;
      });
      toast.error(`Failed to process ${file.name}: ${errorMessage}`);
    }
  };

  /**
   * Poll the results endpoint until we get a result or timeout
   */
  const pollForResult = async (
    fileKey: string,
    requestId: string,
    file: File,
    previewUrl: string,
    startTime: number
  ) => {
    const pollIntervalMs = 2000; // Poll every 2 seconds
    const maxTries = 30; // Max 30 tries (~1 minute timeout)
    let tries = 0;

    while (tries < maxTries) {
      tries++;
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

      try {
        const res = await fetch(
          `${RESULTS_BASE_URL}?request_id=${encodeURIComponent(requestId)}`
        );

        if (!res.ok) {
          console.error("Results API error:", await res.text());
          continue;
        }

        const results: InferenceResult[] = await res.json();

        if (results && results.length > 0) {
          // Calculate end-to-end latency
          const e2eLatencyMs = Date.now() - startTime;

          // Found result - update state with E2E latency
          setImageStates((prev) => {
            const next = new Map(prev);
            next.set(fileKey, {
              file,
              status: "completed",
              requestId,
              result: results[0],
              previewUrl,
              startTime,
              e2eLatencyMs,
            });
            return next;
          });
          toast.success(`Inference complete for ${file.name}`);
          return;
        }
      } catch (error) {
        console.error("Polling error:", error);
      }
    }

    // Timeout reached
    setImageStates((prev) => {
      const next = new Map(prev);
      next.set(fileKey, {
        file,
        status: "timeout",
        requestId,
        error: "Inference timed out",
        previewUrl,
        startTime,
      });
      return next;
    });
    toast.warning(`Inference timed out for ${file.name}`);
  };

  /**
   * Start processing all selected files
   */
  const handleUploadAndInfer = async () => {
    if (selectedFiles.length === 0) {
      toast.error("Please select images first");
      return;
    }

    setIsUploading(true);

    // Create unique keys and generate preview URLs for each file
    const fileEntries = await Promise.all(
      selectedFiles.map(async (file, index) => ({
        file,
        key: `${Date.now()}-${index}-${file.name}`,
        previewUrl: await fileToDataUrl(file),
      }))
    );

    // Initialize all files as pending with preview URLs
    setImageStates((prev) => {
      const next = new Map(prev);
      fileEntries.forEach(({ file, key, previewUrl }) => {
        next.set(key, { file, status: "pending", previewUrl });
      });
      return next;
    });

    // Clear selected files since we're now processing them
    setSelectedFiles([]);

    // Process all files concurrently
    await Promise.all(
      fileEntries.map(({ file, key, previewUrl }) => 
        processImage(file, key, previewUrl)
      )
    );

    setIsUploading(false);
  };

  // --------------------------------------------------------------------------
  // IMAGE PREVIEW HANDLER
  // --------------------------------------------------------------------------

  const openImagePreview = (state: ImageState) => {
    if (state.previewUrl) {
      setPreviewImage({
        url: state.previewUrl,
        name: state.file.name,
        prediction: state.result?.predicted_label,
      });
    }
  };

  // --------------------------------------------------------------------------
  // COMPUTED VALUES
  // --------------------------------------------------------------------------

  const completedCount = Array.from(imageStates.values()).filter(
    (s) => s.status === "completed"
  ).length;
  const totalProcessing = imageStates.size;
  const progressPercent =
    totalProcessing > 0 ? (completedCount / totalProcessing) * 100 : 0;

  // Convert imageStates to sorted array for display
  const imageStateArray = Array.from(imageStates.entries()).sort(
    ([a], [b]) => {
      // Sort by timestamp (newest first)
      const timeA = parseInt(a.split("-")[0]);
      const timeB = parseInt(b.split("-")[0]);
      return timeB - timeA;
    }
  );

  // --------------------------------------------------------------------------
  // RENDER HELPER: Status badge for each image
  // --------------------------------------------------------------------------

  const renderStatusBadge = (status: ImageStatus) => {
    switch (status) {
      case "pending":
        return (
          <Badge variant="secondary" className="gap-1">
            <Clock className="h-3 w-3" />
            Pending
          </Badge>
        );
      case "uploading":
        return (
          <Badge className="gap-1 bg-blue-500 hover:bg-blue-600">
            <Loader2 className="h-3 w-3 animate-spin" />
            Uploading
          </Badge>
        );
      case "polling":
        return (
          <Badge className="gap-1 bg-amber-500 hover:bg-amber-600">
            <Loader2 className="h-3 w-3 animate-spin" />
            Processing
          </Badge>
        );
      case "completed":
        return (
          <Badge className="gap-1 bg-emerald-500 hover:bg-emerald-600">
            <CheckCircle2 className="h-3 w-3" />
            Complete
          </Badge>
        );
      case "error":
        return (
          <Badge variant="destructive" className="gap-1">
            <AlertCircle className="h-3 w-3" />
            Error
          </Badge>
        );
      case "timeout":
        return (
          <Badge variant="destructive" className="gap-1">
            <Clock className="h-3 w-3" />
            Timeout
          </Badge>
        );
    }
  };

  // --------------------------------------------------------------------------
  // RENDER
  // --------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950">
      {/* Decorative background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-400/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-6xl mx-auto px-4 py-12">
        {/* Header Section */}
        <header className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl shadow-lg shadow-blue-500/25">
              <Globe className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-slate-900 via-blue-800 to-indigo-900 dark:from-white dark:via-blue-200 dark:to-indigo-200 bg-clip-text text-transparent mb-3">
            AWS Serverless ML Inference
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Upload images to classify them using a serverless machine learning
            pipeline powered by AWS Lambda, S3, API Gateway, RDS, and ECR.
          </p>

          {/* Tech Stack Pills */}
          <div className="flex flex-wrap justify-center gap-2 mt-6">
            {[
              { icon: Server, label: "Lambda" },
              { icon: CloudUpload, label: "S3" },
              { icon: Zap, label: "API Gateway" },
              { icon: Database, label: "RDS" },
              { icon: Container, label: "ECR" },
            ].map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/60 dark:bg-slate-800/60 backdrop-blur-sm rounded-full text-sm font-medium text-slate-700 dark:text-slate-300 border border-slate-200/50 dark:border-slate-700/50"
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </div>
            ))}
          </div>
        </header>

        {/* Upload Card */}
        <Card className="mb-8 border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5 text-blue-600" />
              Image Upload
            </CardTitle>
            <CardDescription>
              Select or drag & drop images to classify. Supports JPEG and PNG
              formats.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`
                relative border-2 border-dashed rounded-xl p-10
                transition-all duration-200 cursor-pointer
                ${
                  isDragActive
                    ? "border-blue-500 bg-blue-50/50 dark:bg-blue-950/50"
                    : "border-slate-300 dark:border-slate-700 hover:border-blue-400 hover:bg-slate-50/50 dark:hover:bg-slate-800/50"
                }
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <div className="flex flex-col items-center gap-4">
                <div
                  className={`
                  p-4 rounded-full transition-colors
                  ${
                    isDragActive
                      ? "bg-blue-100 dark:bg-blue-900"
                      : "bg-slate-100 dark:bg-slate-800"
                  }
                `}
                >
                  <CloudUpload
                    className={`h-10 w-10 ${
                      isDragActive
                        ? "text-blue-600"
                        : "text-slate-400 dark:text-slate-500"
                    }`}
                  />
                </div>
                <div className="text-center">
                  <p className="text-lg font-medium text-slate-700 dark:text-slate-300">
                    {isDragActive ? "Drop images here" : "Drop images here or click to browse"}
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                    PNG, JPG up to 10MB each
                  </p>
                </div>
              </div>
            </div>

            {/* Selected Files List */}
            {selectedFiles.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-slate-700 dark:text-slate-300">
                    Selected Files ({selectedFiles.length})
                  </h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearAllFiles}
                    className="text-slate-500 hover:text-red-600"
                  >
                    Clear All
                  </Button>
                </div>
                <div className="grid gap-2 max-h-48 overflow-y-auto pr-2">
                  {selectedFiles.map((file, index) => (
                    <div
                      key={`${file.name}-${index}`}
                      className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg group"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <FileImage className="h-5 w-5 text-blue-500 shrink-0" />
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-slate-700 dark:text-slate-300 truncate">
                            {file.name}
                          </p>
                          <p className="text-xs text-slate-500">
                            {formatFileSize(file.size)}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(index)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Button */}
            <Button
              onClick={handleUploadAndInfer}
              disabled={selectedFiles.length === 0 || isUploading}
              size="lg"
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg shadow-blue-500/25 transition-all duration-200"
            >
              {isUploading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-5 w-5" />
                  Upload & Infer
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Progress Bar (when processing) */}
        {totalProcessing > 0 && completedCount < totalProcessing && (
          <Card className="mb-8 border-0 shadow-lg bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
            <CardContent className="py-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Processing Images
                </span>
                <span className="text-sm text-slate-500">
                  {completedCount} / {totalProcessing} complete
                </span>
              </div>
              <Progress value={progressPercent} className="h-2" />
            </CardContent>
          </Card>
        )}

        {/* Results Table */}
        {imageStateArray.length > 0 && (
          <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5 text-indigo-600" />
                Inference Results
              </CardTitle>
              <CardDescription>
                Classification results for uploaded images. Click on an image name to preview it.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-lg border overflow-hidden overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-slate-50/50 dark:bg-slate-800/50">
                      <TableHead className="font-semibold">Image</TableHead>
                      <TableHead className="font-semibold">Status</TableHead>
                      <TableHead className="font-semibold">
                        Request ID
                      </TableHead>
                      <TableHead className="font-semibold">
                        Prediction
                      </TableHead>
                      <TableHead className="font-semibold text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Timer className="h-3.5 w-3.5" />
                          E2E Latency
                        </div>
                      </TableHead>
                      <TableHead className="font-semibold text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Zap className="h-3.5 w-3.5" />
                          Infer Latency
                        </div>
                      </TableHead>
                      <TableHead className="font-semibold">
                        Created At
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {imageStateArray.map(([key, state]) => (
                      <TableRow key={key}>
                        <TableCell className="font-medium">
                          <button
                            onClick={() => openImagePreview(state)}
                            className="flex items-center gap-2 hover:text-blue-600 dark:hover:text-blue-400 transition-colors group text-left"
                          >
                            {state.previewUrl ? (
                              <div className="relative w-10 h-10 rounded-md overflow-hidden shrink-0 border border-slate-200 dark:border-slate-700">
                                <img
                                  src={state.previewUrl}
                                  alt={state.file.name}
                                  className="w-full h-full object-cover"
                                />
                                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center">
                                  <Eye className="h-4 w-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                                </div>
                              </div>
                            ) : (
                              <FileImage className="h-4 w-4 text-slate-400 shrink-0" />
                            )}
                            <span className="truncate max-w-[140px] underline-offset-2 group-hover:underline">
                              {state.file.name}
                            </span>
                          </button>
                        </TableCell>
                        <TableCell>{renderStatusBadge(state.status)}</TableCell>
                        <TableCell>
                          <span className="font-mono text-xs text-slate-500 truncate block max-w-[180px]">
                            {state.requestId || "—"}
                          </span>
                        </TableCell>
                        <TableCell>
                          {state.result?.predicted_label ? (
                            <Badge
                              variant="outline"
                              className="bg-indigo-50 dark:bg-indigo-950 border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300"
                            >
                              {state.result.predicted_label}
                            </Badge>
                          ) : state.error ? (
                            <span className="text-sm text-red-500">
                              {state.error}
                            </span>
                          ) : (
                            <span className="text-slate-400">—</span>
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          {state.e2eLatencyMs != null ? (
                            <span className="font-mono text-sm text-blue-600 dark:text-blue-400">
                              {Math.round(state.e2eLatencyMs)} ms
                            </span>
                          ) : (
                            <span className="text-slate-400">—</span>
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          {state.result?.infer_latency_ms != null ? (
                            <span className="font-mono text-sm text-emerald-600 dark:text-emerald-400">
                              {Math.round(state.result.infer_latency_ms)} ms
                            </span>
                          ) : (
                            <span className="text-slate-400">—</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-slate-500">
                            {state.result?.created_at || "—"}
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Footer */}
        <footer className="mt-12 text-center text-sm text-slate-500 dark:text-slate-400">
          <p>
            AWS Serverless ML Inference Demo • Powered by Lambda, S3, API Gateway, RDS & ECR
          </p>
        </footer>
      </div>

      {/* Image Preview Dialog */}
      <Dialog open={!!previewImage} onOpenChange={() => setPreviewImage(null)}>
        <DialogContent className="max-w-3xl max-h-[90vh] p-0 overflow-hidden">
          <DialogHeader className="p-6 pb-0">
            <DialogTitle className="flex items-center gap-3">
              <FileImage className="h-5 w-5 text-blue-600" />
              <span className="truncate">{previewImage?.name}</span>
              {previewImage?.prediction && (
                <Badge
                  variant="outline"
                  className="ml-auto bg-indigo-50 dark:bg-indigo-950 border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300"
                >
                  {previewImage.prediction}
                </Badge>
              )}
            </DialogTitle>
          </DialogHeader>
          <div className="p-6 pt-4">
            {previewImage?.url && (
              <div className="rounded-lg overflow-hidden bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                <img
                  src={previewImage.url}
                  alt={previewImage.name}
                  className="max-w-full max-h-[60vh] object-contain"
                />
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
