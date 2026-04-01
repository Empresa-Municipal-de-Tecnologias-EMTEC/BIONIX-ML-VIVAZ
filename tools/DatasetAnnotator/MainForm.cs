using System.Drawing.Drawing2D;
using System.Globalization;

namespace DatasetAnnotator;

public class MainForm : Form
{
    // ── Dataset state ─────────────────────────────────────────────────────────
    private readonly List<string> allImages = new();
    private int currentIndex = -1;
    private Bitmap? currentBitmap;

    // ── Annotation state ──────────────────────────────────────────────────────
    private PointF? normPt1;      // normalized [0,1] top-left corner
    private PointF? normPt2;      // normalized [0,1] bottom-right corner
    private int clickPhase;       // 0=idle, 1=first point placed, 2=box defined
    private bool boxModified;
    private bool clearBoxRequested;
    private Point? liveMousePos;

    // ── UI controls ───────────────────────────────────────────────────────────
    private readonly Button btnOpenFolder = new();
    private readonly NumericUpDown nudDepth = new();
    private readonly Label lblCounter = new();
    private readonly PictureBox pictureBox = new();
    private readonly Button btnPrev = new();
    private readonly Button btnNext = new();
    private readonly Button btnClearBox = new();
    private readonly Label lblHint = new();
    private readonly Label lblStatus = new();
    private readonly Label lblBoxCoords = new();
    private readonly Label lblModified = new();

    public MainForm()
    {
        Text = "Dataset Annotator  —  Face Box Editor";
        Size = new Size(1000, 740);
        MinimumSize = new Size(640, 480);
        BackColor = Color.FromArgb(30, 30, 30);
        ForeColor = Color.White;
        KeyPreview = true;
        KeyDown += OnKeyDown;
        BuildUI();
    }

    // ── UI construction ───────────────────────────────────────────────────────

    private void BuildUI()
    {
        // ── Top toolbar ───────────────────────────────────────────────────────
        var top = new Panel
        {
            Dock = DockStyle.Top,
            Height = 46,
            BackColor = Color.FromArgb(48, 48, 48),
            Padding = new Padding(6, 6, 6, 0)
        };

        Style(btnOpenFolder, "📁  Abrir Dataset", 150, 32, Color.FromArgb(55, 115, 190));
        btnOpenFolder.Location = new Point(6, 7);
        btnOpenFolder.Click += (_, _) => OpenDataset();

        var lblD = MakeLabel("Profundidade:", new Point(164, 13));

        nudDepth.Minimum = 1; nudDepth.Maximum = 20; nudDepth.Value = 5;
        nudDepth.Size = new Size(58, 26);
        nudDepth.Location = new Point(262, 10);
        nudDepth.BackColor = Color.FromArgb(60, 60, 60);
        nudDepth.ForeColor = Color.White;

        var lblDHelp = MakeLabel("(subníveis)", new Point(325, 13), Color.Gray);

        lblCounter.AutoSize = true;
        lblCounter.Location = new Point(420, 13);
        lblCounter.ForeColor = Color.LightGray;
        lblCounter.Text = "Nenhuma imagem carregada";

        top.Controls.AddRange(new Control[] { btnOpenFolder, lblD, nudDepth, lblDHelp, lblCounter });
        Controls.Add(top);

        // ── Bottom bar ────────────────────────────────────────────────────────
        var bot = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 72,
            BackColor = Color.FromArgb(48, 48, 48),
            Padding = new Padding(6)
        };

        Style(btnPrev, "◀  Anterior   [←]", 140, 38, Color.FromArgb(68, 68, 68));
        Style(btnNext, "Próxima  [→]  ▶", 140, 38, Color.FromArgb(55, 115, 190));
        Style(btnClearBox, "✕  Limpar box  [Del]", 140, 38, Color.FromArgb(155, 50, 50));

        btnPrev.Location = new Point(6, 16);
        btnNext.Location = new Point(152, 16);
        btnClearBox.Location = new Point(298, 16);

        btnPrev.Click += (_, _) => Navigate(-1);
        btnNext.Click += (_, _) => Navigate(+1);
        btnClearBox.Click += OnClearBox;

        // Info area on the right
        lblHint.AutoSize = true; lblHint.ForeColor = Color.LightYellow; lblHint.Font = new Font("Segoe UI", 9f);
        lblHint.Location = new Point(448, 8);

        lblBoxCoords.AutoSize = true; lblBoxCoords.ForeColor = Color.LightGreen; lblBoxCoords.Font = new Font("Segoe UI", 8.5f);
        lblBoxCoords.Location = new Point(448, 28);

        lblModified.AutoSize = true; lblModified.ForeColor = Color.Orange; lblModified.Font = new Font("Segoe UI", 8f, FontStyle.Bold);
        lblModified.Location = new Point(448, 48);

        lblStatus.AutoSize = false; lblStatus.Size = new Size(400, 16); lblStatus.ForeColor = Color.DimGray;
        lblStatus.Font = new Font("Segoe UI", 8f); lblStatus.Location = new Point(750, 28);
        lblStatus.TextAlign = ContentAlignment.MiddleRight;
        lblStatus.Anchor = AnchorStyles.Right | AnchorStyles.Bottom;

        bot.Controls.AddRange(new Control[] { btnPrev, btnNext, btnClearBox,
                                               lblHint, lblBoxCoords, lblModified, lblStatus });
        Controls.Add(bot);

        // ── Picture box ───────────────────────────────────────────────────────
        pictureBox.Dock = DockStyle.Fill;
        pictureBox.BackColor = Color.Black;
        pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
        pictureBox.Cursor = Cursors.Cross;
        pictureBox.MouseClick += OnPictureClick;
        pictureBox.MouseMove  += OnPictureMove;
        pictureBox.Paint      += OnPicturePaint;
        Controls.Add(pictureBox);
    }

    private static void Style(Button b, string text, int w, int h, Color back)
    {
        b.Text = text; b.Size = new Size(w, h);
        b.BackColor = back; b.ForeColor = Color.White;
        b.FlatStyle = FlatStyle.Flat;
        b.FlatAppearance.BorderSize = 1;
        b.FlatAppearance.BorderColor = Color.FromArgb(80, 80, 80);
        b.Font = new Font("Segoe UI", 9f);
    }

    private static Label MakeLabel(string text, Point loc, Color? color = null)
    {
        var lbl = new Label { Text = text, AutoSize = true, Location = loc };
        lbl.ForeColor = color ?? Color.Silver;
        return lbl;
    }

    // ── Dataset loading ───────────────────────────────────────────────────────

    private void OpenDataset()
    {
        using var dlg = new FolderBrowserDialog
        {
            Description = "Selecione o diretório raiz do dataset",
            UseDescriptionForTitle = true
        };
        if (dlg.ShowDialog() != DialogResult.OK) return;

        allImages.Clear();
        ScanDir(dlg.SelectedPath, 0, (int)nudDepth.Value);
        allImages.Sort(StringComparer.OrdinalIgnoreCase);

        if (allImages.Count == 0)
        {
            lblCounter.Text = "Nenhuma imagem encontrada neste diretório!";
            pictureBox.Image = null;
            currentBitmap?.Dispose(); currentBitmap = null;
            return;
        }

        currentIndex = 0;
        LoadCurrent();
    }

    private static readonly string[] ImgExts = { "*.bmp", "*.jpg", "*.jpeg", "*.png" };

    private void ScanDir(string dir, int depth, int maxDepth)
    {
        try
        {
            foreach (var ext in ImgExts)
                allImages.AddRange(Directory.GetFiles(dir, ext));

            if (depth < maxDepth)
                foreach (var sub in Directory.GetDirectories(dir))
                    ScanDir(sub, depth + 1, maxDepth);
        }
        catch { /* skip inaccessible dirs */ }
    }

    // ── Navigation ────────────────────────────────────────────────────────────

    private void Navigate(int delta)
    {
        if (allImages.Count == 0) return;
        CommitBox();
        currentIndex = Math.Clamp(currentIndex + delta, 0, allImages.Count - 1);
        LoadCurrent();
    }

    private void LoadCurrent()
    {
        if (currentIndex < 0 || currentIndex >= allImages.Count) return;

        var path = allImages[currentIndex];

        currentBitmap?.Dispose();
        currentBitmap = null;

        try
        {
            // Open via stream to avoid GDI file lock
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            currentBitmap = new Bitmap(fs);
        }
        catch (Exception ex)
        {
            lblStatus.Text = "Erro ao carregar: " + ex.Message;
            return;
        }

        // Reset drawing state
        normPt1 = null; normPt2 = null;
        clickPhase = 0; boxModified = false; clearBoxRequested = false;
        liveMousePos = null;

        // Load existing .box annotation
        var boxPath = Path.ChangeExtension(path, ".box");
        if (File.Exists(boxPath))
        {
            try
            {
                var parts = File.ReadAllText(boxPath).Trim()
                    .Split(new[] { ' ', '\t', ',' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4)
                {
                    float x0 = float.Parse(parts[0], CultureInfo.InvariantCulture);
                    float y0 = float.Parse(parts[1], CultureInfo.InvariantCulture);
                    float x1 = float.Parse(parts[2], CultureInfo.InvariantCulture);
                    float y1 = float.Parse(parts[3], CultureInfo.InvariantCulture);

                    // Pixel coords (> 1.5) → normalize
                    if (x0 > 1.5f || y0 > 1.5f || x1 > 1.5f || y1 > 1.5f)
                    {
                        float iw = Math.Max(1, currentBitmap.Width  - 1);
                        float ih = Math.Max(1, currentBitmap.Height - 1);
                        x0 /= iw; y0 /= ih; x1 /= iw; y1 /= ih;
                    }

                    normPt1 = new PointF(Math.Min(x0, x1), Math.Min(y0, y1));
                    normPt2 = new PointF(Math.Max(x0, x1), Math.Max(y0, y1));
                    clickPhase = 2;
                }
            }
            catch { /* ignore malformed .box */ }
        }

        pictureBox.Image = currentBitmap;
        lblCounter.Text = $"{currentIndex + 1} / {allImages.Count}   |   {Path.GetFileName(path)}   [{Path.GetDirectoryName(path)}]";
        lblStatus.Text = path;
        UpdateHint(); UpdateBoxLabel(); UpdateModifiedLabel();
        pictureBox.Invalidate();
    }

    // ── Box persistence ───────────────────────────────────────────────────────

    private void CommitBox(bool force = false)
    {
        if ((!boxModified && !force) || currentIndex < 0) return;

        var boxPath = Path.ChangeExtension(allImages[currentIndex], ".box");

        try
        {
            if (clearBoxRequested || !normPt1.HasValue || !normPt2.HasValue)
            {
                if (File.Exists(boxPath)) File.Delete(boxPath);
                lblModified.Text = "box removida";
                lblModified.ForeColor = Color.Gray;
            }
            else
            {
                float x0 = Math.Min(normPt1.Value.X, normPt2.Value.X);
                float y0 = Math.Min(normPt1.Value.Y, normPt2.Value.Y);
                float x1 = Math.Max(normPt1.Value.X, normPt2.Value.X);
                float y1 = Math.Max(normPt1.Value.Y, normPt2.Value.Y);

                if ((x1 - x0) > 0.002f && (y1 - y0) > 0.002f)
                {
                    File.WriteAllText(boxPath, $"{x0:F6} {y0:F6} {x1:F6} {y1:F6}");
                    lblModified.Text = $"✔ salvo: {Path.GetFileName(boxPath)}";
                    lblModified.ForeColor = Color.LightGreen;
                }
                else
                {
                    lblModified.Text = "caixa muito pequena — não salva";
                    lblModified.ForeColor = Color.Orange;
                }
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Erro ao salvar {boxPath}:\n{ex.Message}",
                "Erro ao salvar", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        boxModified = false;
        clearBoxRequested = false;
    }

    private void OnClearBox(object? sender, EventArgs e)
    {
        normPt1 = null; normPt2 = null;
        clickPhase = 0; boxModified = true; clearBoxRequested = true;
        liveMousePos = null;
        UpdateHint(); UpdateBoxLabel(); UpdateModifiedLabel();
        pictureBox.Invalidate();
        // Save immediately (will delete the .box file)
        CommitBox(force: true);
    }

    // ── Mouse events ──────────────────────────────────────────────────────────

    private void OnPictureClick(object? sender, MouseEventArgs e)
    {
        if (currentBitmap == null || e.Button != MouseButtons.Left) return;

        var norm = ScreenToNorm(e.Location);

        if (clickPhase == 0 || clickPhase == 2)
        {
            // Start a new box
            normPt1 = norm; normPt2 = null;
            clickPhase = 1;
        }
        else // clickPhase == 1 → complete the box
        {
            normPt2 = norm;
            clickPhase = 2;
            boxModified = true;
            clearBoxRequested = false;
            UpdateBoxLabel();
            // Save immediately — don't wait for navigation
            CommitBox(force: true);
        }

        UpdateHint(); UpdateModifiedLabel();
        pictureBox.Invalidate();
    }

    private void OnPictureMove(object? sender, MouseEventArgs e)
    {
        liveMousePos = e.Location;
        if (clickPhase == 1) pictureBox.Invalidate(); // rubber-band refresh
    }

    // ── Drawing ───────────────────────────────────────────────────────────────

    private void OnPicturePaint(object? sender, PaintEventArgs e)
    {
        if (currentBitmap == null) return;

        var g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        var imgRect = GetImageRect();
        if (imgRect.Width == 0) return;

        // Rubber-band preview while placing second point
        if (clickPhase == 1 && normPt1.HasValue && liveMousePos.HasValue)
        {
            var liveNorm = ScreenToNorm(liveMousePos.Value);
            DrawBox(g, imgRect, normPt1.Value, liveNorm, Color.Yellow, dashed: true);
            DrawDot(g, imgRect, normPt1.Value, Color.Yellow, radius: 5);
        }

        // Committed box (loaded or newly defined)
        if (clickPhase == 2 && normPt1.HasValue && normPt2.HasValue)
        {
            var color = boxModified ? Color.Lime : Color.DeepSkyBlue;
            DrawBox(g, imgRect, normPt1.Value, normPt2.Value, color, dashed: false);
            DrawDot(g, imgRect, normPt1.Value, color, radius: 4);
            DrawDot(g, imgRect, normPt2.Value, color, radius: 4);
        }
    }

    private static void DrawBox(Graphics g, Rectangle imgRect,
                                  PointF n1, PointF n2, Color color, bool dashed)
    {
        float x0 = imgRect.X + n1.X * imgRect.Width;
        float y0 = imgRect.Y + n1.Y * imgRect.Height;
        float x1 = imgRect.X + n2.X * imgRect.Width;
        float y1 = imgRect.Y + n2.Y * imgRect.Height;
        float rx = Math.Min(x0, x1), ry = Math.Min(y0, y1);
        float rw = Math.Abs(x1 - x0), rh = Math.Abs(y1 - y0);

        // Shadow for contrast on any background
        using var shadow = new Pen(Color.Black, 4);
        g.DrawRectangle(shadow, rx - 1, ry - 1, rw + 2, rh + 2);

        using var pen = new Pen(color, 2);
        if (dashed) pen.DashStyle = DashStyle.Dash;
        g.DrawRectangle(pen, rx, ry, rw, rh);
    }

    private static void DrawDot(Graphics g, Rectangle imgRect, PointF norm, Color color, int radius)
    {
        float x = imgRect.X + norm.X * imgRect.Width;
        float y = imgRect.Y + norm.Y * imgRect.Height;
        using var brush = new SolidBrush(color);
        g.FillEllipse(brush, x - radius, y - radius, radius * 2, radius * 2);
        using var pen = new Pen(Color.Black, 1);
        g.DrawEllipse(pen, x - radius, y - radius, radius * 2, radius * 2);
    }

    // ── Coordinate helpers ────────────────────────────────────────────────────

    /// Returns the rectangle within the PictureBox where the image is rendered (Zoom mode).
    private Rectangle GetImageRect()
    {
        if (currentBitmap == null) return Rectangle.Empty;
        var pb = pictureBox.ClientSize;
        float ia = (float)currentBitmap.Width / currentBitmap.Height;
        float pa = (float)pb.Width / pb.Height;
        int w, h;
        if (ia > pa) { w = pb.Width;  h = (int)(pb.Width  / ia); }
        else         { h = pb.Height; w = (int)(pb.Height * ia); }
        return new Rectangle((pb.Width - w) / 2, (pb.Height - h) / 2, w, h);
    }

    /// Converts a screen point (relative to pictureBox) to normalized image coordinates [0,1].
    private PointF ScreenToNorm(Point p)
    {
        var r = GetImageRect();
        if (r.Width == 0 || r.Height == 0) return PointF.Empty;
        return new PointF(
            Math.Clamp((float)(p.X - r.X) / r.Width,  0f, 1f),
            Math.Clamp((float)(p.Y - r.Y) / r.Height, 0f, 1f));
    }

    // ── UI label helpers ──────────────────────────────────────────────────────

    private void UpdateHint() => lblHint.Text = clickPhase switch
    {
        0 => "Clique no 1º canto do rosto  (ou [←][→] para navegar)",
        1 => "Clique no canto oposto do rosto para finalizar o retângulo",
        _ => "Box salva ✔  \u00b7  Clique para redesenhar  \u00b7  [Del] para apagar  \u00b7  [Ctrl+S] salvar"
    };

    private void UpdateBoxLabel()
    {
        if (!normPt1.HasValue || !normPt2.HasValue) { lblBoxCoords.Text = "Sem box"; return; }
        float x0 = Math.Min(normPt1.Value.X, normPt2.Value.X);
        float y0 = Math.Min(normPt1.Value.Y, normPt2.Value.Y);
        float x1 = Math.Max(normPt1.Value.X, normPt2.Value.X);
        float y1 = Math.Max(normPt1.Value.Y, normPt2.Value.Y);
        lblBoxCoords.Text = $"box: {x0:F3} {y0:F3} → {x1:F3} {y1:F3}   "
                          + $"({(x1 - x0) * 100:F1}% × {(y1 - y0) * 100:F1}% da imagem)";
    }

    private void UpdateModifiedLabel()
    {
        lblModified.Text = boxModified ? "● modificado (será salvo ao avançar)" : "";
    }

    // ── Keyboard ──────────────────────────────────────────────────────────────

    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        switch (e.KeyCode)
        {
            case Keys.Right: case Keys.D: Navigate(+1); e.Handled = true; break;
            case Keys.Left:  case Keys.A: Navigate(-1); e.Handled = true; break;
            // Only Delete clears the box — Backspace is intentionally excluded
            // so users don't accidentally wipe annotations when navigating by keyboard
            case Keys.Delete: OnClearBox(null, EventArgs.Empty); e.Handled = true; break;
            case Keys.S when e.Control: CommitBox(force: true); e.Handled = true; break;
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    protected override void OnFormClosing(FormClosingEventArgs e)
    { CommitBox(); base.OnFormClosing(e); }

    protected override void Dispose(bool disposing)
    { if (disposing) currentBitmap?.Dispose(); base.Dispose(disposing); }
}
