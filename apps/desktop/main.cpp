#include "qwen3_cpu.hpp"
#include "tokenizer/qwen3_tokenizer.hpp"

#include <QApplication>
#include <QAtomicInteger>
#include <QFont>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QScrollBar>
#include <QLabel>
#include <QMetaObject>
#include <QPushButton>
#include <QTextBrowser>
#include <QTextEdit>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Lives entirely on the inference thread. Only cancel() is called across
// threads, and it touches an atomic flag.
class ChatWorker final : public QObject
{
    Q_OBJECT
public:
    explicit ChatWorker(QString modelPath) : path_(std::move(modelPath)) {}
    void cancel() noexcept { cancelled_.store(true); }

public slots:
    void load()
    {
        try
        {
            tokenizer_ = std::make_unique<Qwen3Tokenizer>((path_ + "/tokenizer.json").toStdString());
            model_ = std::make_unique<Qwen3Cpu>(Qwen3Cpu::fromDirectory(path_.toStdString()));
            emit loaded();
        }
        catch (const std::exception &e) { emit failure(QString::fromUtf8(e.what())); }
    }

    void newChat() { history_.clear(); }

    void send(QString input)
    {
        if (!model_ || !tokenizer_)
        {
            emit failure("The model is not loaded.");
            emit finished(false);
            return;
        }
        cancelled_.store(false);
        auto messages = history_;
        messages.emplace_back("user", input.toStdString());
        try
        {
            const auto prompt = tokenizer_->chatConversation(messages);
            std::vector<std::size_t> generated;
            const auto tokens = model_->generateStreaming(prompt, 512, [&](std::size_t id) {
                if (cancelled_.load()) return false;
                if (id != 151643 && id != 151645)
                {
                    generated.push_back(id);
                    // Decode the full generated sequence so UTF-8 bytes that
                    // span multiple tokens are handled safely.
                    emit partial(QString::fromStdString(tokenizer_->decode(generated)));
                }
                return !cancelled_.load();
            });
            (void)tokens;
            // Keep a stopped response if any text was produced, so the
            // next user message has an accurate conversation history.
            const auto response = tokenizer_->decode(generated);
            if (!response.empty())
            {
                history_ = std::move(messages);
                history_.emplace_back("assistant", response);
            }
            emit finished(cancelled_.load());
        }
        catch (const std::exception &e)
        {
            emit failure(QString::fromUtf8(e.what()));
            emit finished(false);
        }
    }

signals:
    void loaded();
    void partial(QString text);
    void finished(bool cancelled);
    void failure(QString error);

private:
    QString path_;
    std::unique_ptr<Qwen3Tokenizer> tokenizer_;
    std::unique_ptr<Qwen3Cpu> model_;
    std::vector<std::pair<std::string, std::string>> history_;
    std::atomic<bool> cancelled_{false};
};

class Composer final : public QTextEdit
{
    Q_OBJECT
public:
    using QTextEdit::QTextEdit;
signals:
    void submitted();
protected:
    void keyPressEvent(QKeyEvent *event) override
    {
        if ((event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            && !(event->modifiers() & Qt::ShiftModifier))
        {
            emit submitted();
            return;
        }
        QTextEdit::keyPressEvent(event);
    }
};

class ChatWindow final : public QWidget
{
    Q_OBJECT
public:
    explicit ChatWindow(const QString &path)
    {
        setWindowTitle("Super Science");
        resize(760, 620);
        setMinimumSize(480, 360);
        setStyleSheet(
            "QWidget {background:#11151c;color:#e2e8f0;font-size:14px;}"
            "QTextBrowser {background:#11151c;border:0;padding:12px;}"
            "QTextEdit {background:#1d2530;border:1px solid #344152;"
            "border-radius:9px;padding:9px;}"
            "QPushButton {background:#293442;border:0;border-radius:7px;"
            "padding:9px 15px;}"
            "QPushButton:hover {background:#38485c;}"
            "QPushButton:disabled {color:#697586;}"
            "QLabel {color:#a8b6c8;}"
        );

        auto *layout = new QVBoxLayout(this);
        layout->setContentsMargins(20, 16, 20, 16);
        layout->setSpacing(12);

        auto *top = new QHBoxLayout();
        auto *title = new QLabel("Super Science");
        QFont heading = title->font();
        heading.setPointSize(15);
        heading.setBold(true);
        title->setFont(heading);
        status_ = new QLabel("Loading model...");
        top->addWidget(title);
        top->addStretch();
        top->addWidget(status_);
        layout->addLayout(top);

        transcript_ = new QTextBrowser();
        transcript_->setOpenExternalLinks(false);
        layout->addWidget(transcript_, 1);

        composer_ = new Composer();
        composer_->setPlaceholderText("Type a message... (Enter to send, Shift+Enter for newline)");
        composer_->setFixedHeight(90);
        composer_->setEnabled(false);
        layout->addWidget(composer_);

        auto *controls = new QHBoxLayout();
        newChat_ = new QPushButton("New chat");
        send_ = new QPushButton("Send");
        stop_ = new QPushButton("Stop");
        newChat_->setEnabled(false);
        send_->setEnabled(false);
        stop_->setEnabled(false);
        controls->addWidget(newChat_);
        controls->addStretch();
        controls->addWidget(stop_);
        controls->addWidget(send_);
        layout->addLayout(controls);

        worker_ = new ChatWorker(path);
        worker_->moveToThread(&thread_);
        connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
        connect(this, &ChatWindow::loadRequested, worker_, &ChatWorker::load);
        connect(this, &ChatWindow::sendRequested, worker_, &ChatWorker::send);
        connect(this, &ChatWindow::newChatRequested, worker_, &ChatWorker::newChat);
        connect(worker_, &ChatWorker::loaded, this, [this]() {
            status_->setText("Ready · Local Qwen3");
            composer_->setEnabled(true);
            newChat_->setEnabled(true);
            send_->setEnabled(true);
            composer_->setFocus();
        });
        connect(worker_, &ChatWorker::failure, this, [this](const QString &message) {
            status_->setText("Error: " + message);
        });
        connect(worker_, &ChatWorker::partial, this, [this](const QString &text) {
            currentReply_ = text;
            render();
        });
        connect(worker_, &ChatWorker::finished, this, [this](bool stopped) {
            busy_ = false;
            status_->setText(stopped ? "Stopped" : "Ready · Local Qwen3");
            composer_->setEnabled(true);
            send_->setEnabled(true);
            newChat_->setEnabled(true);
            stop_->setEnabled(false);
            if (!currentReply_.isEmpty())
                transcriptItems_.emplace_back("Super Science", currentReply_);
            currentReply_.clear();
            render();
            composer_->setFocus();
        });
        connect(send_, &QPushButton::clicked, this, &ChatWindow::submit);
        connect(composer_, &Composer::submitted, this, &ChatWindow::submit);
        connect(stop_, &QPushButton::clicked, this, [this]() {
            worker_->cancel();
            status_->setText("Stopping...");
        });
        connect(newChat_, &QPushButton::clicked, this, [this]() {
            if (busy_) return;
            transcriptItems_.clear();
            currentReply_.clear();
            render();
            emit newChatRequested();
        });

        thread_.start();
        emit loadRequested();
    }

    ~ChatWindow() override
    {
        worker_->cancel();
        thread_.quit();
        thread_.wait();
    }

signals:
    void loadRequested();
    void sendRequested(QString text);
    void newChatRequested();

private slots:
    void submit()
    {
        if (busy_) return;
        const auto input = composer_->toPlainText().trimmed();
        if (input.isEmpty()) return;
        busy_ = true;
        currentReply_.clear();
        transcriptItems_.emplace_back("You", input);
        composer_->clear();
        composer_->setEnabled(false);
        send_->setEnabled(false);
        newChat_->setEnabled(false);
        stop_->setEnabled(true);
        status_->setText("Generating...");
        render();
        emit sendRequested(input);
    }

private:
    void render()
    {
        QString html;
        for (const auto &item : transcriptItems_)
        {
            html += "<p style='margin:14px 0'><b>" + item.first.toHtmlEscaped()
                + "</b><br>" + item.second.toHtmlEscaped().replace("\n", "<br>") + "</p>";
        }
        if (busy_ && !currentReply_.isEmpty())
            html += "<p style='margin:14px 0'><b>Super Science</b><br>"
                 + currentReply_.toHtmlEscaped().replace("\n", "<br>") + "</p>";
        transcript_->setHtml(html);
        auto *scrollbar = transcript_->verticalScrollBar();
        scrollbar->setValue(scrollbar->maximum());
    }

    QThread thread_;
    ChatWorker *worker_ = nullptr;
    QLabel *status_ = nullptr;
    QTextBrowser *transcript_ = nullptr;
    Composer *composer_ = nullptr;
    QPushButton *send_ = nullptr;
    QPushButton *stop_ = nullptr;
    QPushButton *newChat_ = nullptr;
    std::vector<std::pair<QString, QString>> transcriptItems_;
    QString currentReply_;
    bool busy_ = false;
};

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    if (argc != 2)
    {
        qCritical("Usage: super_science_desktop <model-directory>");
        return 2;
    }
    ChatWindow window(QString::fromLocal8Bit(argv[1]));
    window.show();
    return app.exec();
}

#include "main.moc"
