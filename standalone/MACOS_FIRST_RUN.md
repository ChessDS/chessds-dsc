# Opening a DSC engine on macOS

These downloads are not Apple-notarized, so macOS blocks them until you allow the download yourself, and some chess GUIs (HIARCS Chess Explorer) give up before a first launch finishes. Only do this for a download you trust. Keep the entire extracted folder together.

1. Extract the ZIP (double-click it).
2. Open Terminal (Applications → Utilities → Terminal), paste the line for your model, press Return, and wait until it prints `bestmove` and returns to the prompt (10 to 20 seconds). The line clears the download quarantine for that folder only and runs the engine once so macOS finishes checking its files; no system setting changes.

- DSC-23M:

  ```
  cd ~/Downloads/dsc-23m-0.1.0-macos-arm64 && xattr -dr com.apple.quarantine . && printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./dsc-23m
  ```

- DSC-23M-FEN:

  ```
  cd ~/Downloads/dsc-23m-fen-0.1.0-macos-arm64 && xattr -dr com.apple.quarantine . && printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./dsc-23m-fen
  ```

- DSC-79M:

  ```
  cd ~/Downloads/dsc-79m-0.1.0-macos-arm64 && xattr -dr com.apple.quarantine . && printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./dsc-79m
  ```

- DSC-79M-FEN:

  ```
  cd ~/Downloads/dsc-79m-fen-0.1.0-macos-arm64 && xattr -dr com.apple.quarantine . && printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./dsc-79m-fen
  ```

- DSC-150M:

  ```
  cd ~/Downloads/dsc-150m-0.1.0-macos-arm64 && xattr -dr com.apple.quarantine . && printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./dsc-150m
  ```

- DSC-150M-FEN:

  ```
  cd ~/Downloads/dsc-150m-fen-0.1.0-macos-arm64 && xattr -dr com.apple.quarantine . && printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./dsc-150m-fen
  ```

   If you extracted the ZIP somewhere other than Downloads, replace the folder after `cd` with yours: type `cd `, drag the folder from Finder into Terminal, press Return, then run the rest of the line from `xattr` on.

3. In your chess GUI (HIARCS: Preferences → Engines → Add), select the executable named like the model, for example `dsc-23m-fen`, inside the extracted folder. The ZIP and the folder itself are not engines.

After this one-time step the engine starts in under a second. Verified on 2026-10-23 with the public DSC-23M-FEN Apple silicon download and HIARCS Chess Explorer 1.11: with the quarantine command alone HIARCS still reported "not a valid UCI engine" (it does not wait for the first launch's library checks); after the full line it accepted the engine.

Apple also offers [System Settings → Privacy & Security → Open Anyway](https://support.apple.com/en-us/102445) after a blocked launch, but that approval covers one file at a time; the folder-level command above covers the bundled runtime libraries as well as the executable.
