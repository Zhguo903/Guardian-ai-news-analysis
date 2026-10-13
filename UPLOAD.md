# Upload to GitHub

Create an empty repository named `Guardian-ai-news-analysis`. Avoid initializing it with a README because this project already includes one.

From this folder:

```bash
git init -b main
git add .
git status --short
git commit -m "Add Guardian AI news analysis"
git remote add origin https://github.com/Zhguo903/Guardian-ai-news-analysis.git
git push -u origin main
```

Replace `Zhguo903` with your account. Inspect the staged file list before committing; `data/raw/`, `data/derived/`, credentials, and local environments are ignored. The supplied ZIP contains only public project files and can also be uploaded through GitHub's web interface.

To publish the pre-rendered report, open repository **Settings → Pages**, choose **Deploy from a branch**, and select **main /docs**. `docs/index.html` is the entry point. A custom domain is optional.

After editing source pages, install Python dependencies and Quarto, then run `bash build.sh`. Commit the refreshed `docs/` folder. Full model reruns require an authorized local corpus; see `data/README.md`.

Choose a repository license before inviting reuse if you want to grant open-source permissions. No license was added automatically.
