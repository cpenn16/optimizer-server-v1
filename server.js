import express from "express";
import cors from "cors";

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(cors({
  origin: ["https://cpenn-dfs.com","https://www.cpenn-dfs.com","http://localhost:5173","http://localhost:3000"],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"],
}));

app.get("/", (_, res) => res.send("OK"));

const ALLOWED = new Set(["trucks","cup","xfinity","nfl"]);

// Generic streaming endpoint: /:product/solve_stream
app.post("/:product/solve_stream", async (req, res) => {
  const { product } = req.params;
  if (!ALLOWED.has(product)) return res.status(404).json({ error: "unknown product", product });

  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  if (res.flushHeaders) res.flushHeaders();

  const write = (obj) => res.write(JSON.stringify(obj) + "\n");

  write({ status: "started", product });

  // <<< send exactly 20 steps
  for (let step = 1; step <= 20; step++) {
    await new Promise(r => setTimeout(r, 120)); // replace with real work
    write({
      product,
      step,
      total: 20,
      progress: step / 20,
      msg: step < 20 ? "working…" : "finalizing…",
    });
  }

  // <<< final line so the client KNOWS we’re done
  write({
    product,
    done: true,
    progress: 1,
    result: { lineups: [], meta: { route: `/${product}/solve_stream` } },
  });

  res.end(); // <<< closes the stream cleanly
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Server listening on :${PORT}`));
