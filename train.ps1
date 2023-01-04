param(
    [Parameter(Mandatory=$true)]
    [String]$data,
    [Parameter(Mandatory=$true)]
    [String]$config,
    [String]$suffix = "",
    [String]$weights = "none",
    [int]$size = 192,
    [int]$batch_size = 64,
    [int]$epochs = 250,
    [int]$test_every = 20,
    [String]$device = "0",
    [String]$hyp = "data/hyp.scratch.tiny.hand.yaml",
    [Switch]$cached = $false,
    [Switch]$export_only = $false
)

$data_name = [System.IO.Path]::GetFileNameWithoutExtension($data)
$config_name = [System.IO.Path]::GetFileNameWithoutExtension($config)

$name = "$config_name$suffix-$size"
$experiment_path = "runs\$data_name"

$cached_param = ""
if ($cached) {
    $cached_param = "--cache-images"
}

if (!$export_only)
{
    Write-Host "training $data_name-$name in $experiment_path..."
    python .\train.py --data "$data" --cfg "$config" --hyp "$hyp" --img-size $size $size --batch-size $batch_size --project "$experiment_path" --name "$name" --weights "$weights" --device $device --epochs $epochs --save_period $test_every --exist-ok
}

if(!$?)
{
    Write-Warning "Training did not complete, please check the output!"
    exit 1
}

Write-Host "exporting..."
$weights_dir = "$experiment_path\$name\weights"
$weight_path = "$weights_dir\best"
python .\models\export.py --weights "$weight_path.pt" --img-size $size $size --batch-size 1 --simplify --grid --export-nms

$onnx_name = "$data_name-$name.onnx"
$onnx_path = "$weights_dir\$onnx_name"

if (Test-Path $onnx_path) {
  Remove-Item -Path $onnx_path
}

Rename-Item -Path "$weight_path.onnx" -NewName "$onnx_name" -Force
Copy-Item -Path "$experiment_path\$name\weights\$onnx_name" -Destination "weights\$onnx_name" -Force
Remove-Item -Path "$weight_path.torchscript.pt" -Force

Write-Host "finished $name"