# Opening a DSC engine on macOS

These downloads are not Apple-notarized. Only override macOS checks for a download you trust. Keep the entire extracted folder together.

1. Extract the ZIP. Open Terminal (Applications → Utilities → Terminal).
2. Type `xattr -dr com.apple.quarantine`, add a space, then drag the **whole extracted folder** from Finder into Terminal. Press Return. This clears download quarantine only for that folder and its contents; it does not turn off Gatekeeper globally.
3. Run the engine once before adding it to a chess GUI. Type the following, add a space after the pipe, then drag the **named executable inside the folder** (for example `dsc-79m-fen`) into Terminal and press Return:

   ```sh
   printf 'uci\nisready\nquit\n' |
   ```

   Wait for `uciok`, then `readyok`, and for the Terminal prompt to return. The first launch can be slow: one tested Mac took about 18 seconds. A GUI may report “not a valid engine” if it times out during that initial launch.
4. In HIARCS, open Preferences → Engines → Add and select that same named executable. The ZIP and the enclosing folder are not engine executables.

Verified on 2026-10-22 with the public DSC-79M-FEN Apple silicon download: the ZIP matched the published checksum; clearing quarantine allowed UCI startup and a legal move; HIARCS then accepted the engine as DSC-79M-FEN.

Apple also provides [System Settings → Privacy & Security → Open Anyway](https://support.apple.com/en-us/102445) after an attempted launch. The folder-level procedure above covers the bundled runtime libraries as well as the executable.
