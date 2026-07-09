import "./app.css";
import { mount } from "svelte";
import { invoke } from "@tauri-apps/api/core";
import App from "./App.svelte";

// Webview console output is invisible in the tauri dev terminal; forward
// every uncaught error and rejection to the Rust side so runtime failures
// (which otherwise present as a silently broken UI) show up in the log.
function reportError(kind: string, message: string) {
  void invoke("log_js_error", { message: `${kind}: ${message}` }).catch(() => {});
}

window.addEventListener("error", (e) => {
  reportError("error", `${e.message} (${e.filename}:${e.lineno})`);
});
window.addEventListener("unhandledrejection", (e) => {
  reportError("unhandledrejection", String(e.reason));
});

const app = mount(App, { target: document.getElementById("app")! });
export default app;
