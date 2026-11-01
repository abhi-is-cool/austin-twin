# Build a CA bundle that Python (requests/ssl) and GDAL (curl) both accept.
#
# This machine runs an HTTPS-inspecting security product, so outbound TLS is
# re-signed by a local root CA that ships in the Windows certificate store but
# not in certifi's bundle. Anything using certifi therefore fails to verify.
# Combining the Windows roots with certifi's gives one bundle that works for
# both the public CAs and the local interception CA.
$ErrorActionPreference = "Stop"

$py = "c:\dev\research\austin-twin\.venv\Scripts\python.exe"
$out = "c:\dev\research\austin-twin\.venv\ca-bundle.pem"

$sb = New-Object System.Text.StringBuilder
$n = 0
Get-ChildItem -Path Cert:\LocalMachine\Root, Cert:\CurrentUser\Root, Cert:\LocalMachine\CA |
    Sort-Object -Property Thumbprint -Unique |
    ForEach-Object {
        $b64 = [Convert]::ToBase64String($_.RawData, 'InsertLineBreaks')
        [void]$sb.AppendLine("# subject: $($_.Subject)")
        [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
        [void]$sb.AppendLine($b64)
        [void]$sb.AppendLine("-----END CERTIFICATE-----")
        $n++
    }
Write-Output "windows store certificates: $n"

$certifi = & $py -c "import certifi;print(certifi.where())"
Write-Output "appending certifi bundle: $certifi"
[void]$sb.AppendLine((Get-Content -Raw $certifi))

Set-Content -Path $out -Value $sb.ToString() -Encoding ascii
Write-Output "wrote $out"
