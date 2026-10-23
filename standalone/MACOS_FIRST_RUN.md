# Opening a DSC engine on macOS

These downloads are not Apple-notarized, so macOS blocks them until you allow the download yourself. Only do this for a download you trust. Keep the entire extracted folder together.

1. Extract the ZIP (double-click it in Downloads).
2. Open Terminal (Applications → Utilities → Terminal), paste the line below with your folder's name, and press Return:

   ```sh
   xattr -dr com.apple.quarantine ~/Downloads/dsc-150m-0.1.0-macos-arm64
   ```

   This clears the download quarantine for that folder only; it does not change any system setting. Without it macOS stops the engine once for the executable and again for each library it loads.
3. In your chess GUI (HIARCS: Preferences → Engines → Add), select the named executable inside the folder, for example `dsc-150m`. The ZIP and the folder itself are not engines.

The engine identifies itself to the GUI immediately. The first time it analyses on a computer, macOS checks the bundled libraries, which takes about 10 to 20 seconds before the first moves appear; every later start takes under a second. If your GUI still reports that the engine is not valid, check that step 2 ran on the extracted folder and not on the ZIP.

Apple also offers [System Settings → Privacy & Security → Open Anyway](https://support.apple.com/en-us/102445) after a blocked launch, but that approval covers one file at a time; the folder-level command above covers the bundled runtime libraries as well as the executable.
