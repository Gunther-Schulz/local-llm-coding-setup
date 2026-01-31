# Investigating whether Cursor uses our context_length

When you use a custom OpenAI base URL (e.g. ngrok → proxy), Cursor may or may not read `context_length` from our `GET /v1/models` response. Here are ways to find out.

## 1. Check if Cursor calls /v1/models

The proxy now logs every `GET /v1/models` request:

```
[INFO] GET /v1/models from <IP>: N model(s), context_length=131072
```

- Restart the proxy, then in Cursor: open the model picker, switch model, or start a new chat.
- Check `logs/proxy.log` (or proxy stdout if DEBUG=1). If you see that line around the same time, Cursor (or its backend) is calling our models endpoint. If you never see it when you only use Cursor, they might cache the list or use a different path.

## 2. Behaviour test: fake context_length

If Cursor uses our `context_length`, changing it should change Cursor’s behaviour (e.g. how much it sends before truncating, or what it shows in the UI).

1. In `proxy/server.py`, in `list_models()`, temporarily force a distinctive value, e.g.:
   ```python
   ctx_limit = 999999  # was: get_effective_context_limit()
   ```
2. Restart the proxy. Use Cursor with your custom model for a bit.
3. Check whether:
   - The UI shows a different context (e.g. “0/999999” or similar), or
   - Cursor allows much longer conversations before truncating or erroring.
4. If behaviour changes, Cursor is using the value. Restore `ctx_limit = get_effective_context_limit()`.

## 3. Inspect Cursor’s network traffic

- **Electron DevTools**  
  In Cursor: Help → Toggle Developer Tools → Network tab. Reload or open model picker / start chat. Look for requests to your base URL (e.g. `https://xxx.ngrok.io/v1/models`). Check if that request exists and when it happens (app start, model switch, etc.).

- **System proxy**  
  Run Cursor behind a proxy (e.g. mitmproxy, Charles) and filter for your ngrok host. Confirm whether and when `GET /v1/models` is called and what response Cursor gets (e.g. does the response include `context_length`).

## 4. Cursor config / stored model data

Cursor may store model metadata (name, context, etc.) under:

- macOS: `~/Library/Application Support/Cursor/`
- Linux: `~/.config/Cursor/` or similar

Search for JSON or config that mentions your model name or base URL and see if there is a `contextLength` or similar field and whether it matches our `context_length` (e.g. 131072). If it matches and you didn’t set it manually, that’s a hint they read it from the API.

---

**Summary:** Start with (1) to see if `/v1/models` is called; then (2) to see if changing `context_length` changes behaviour. Use (3) and (4) if you want more detail.
