import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 الدردشة الذكية مع المستندات")
st.caption("نظام RAG للإجابة عن الأسئلة من محتوى ملفات PDF")


# =========================================================
# تهيئة Session State
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("⚙️ إعدادات النظام")

api_key = st.sidebar.text_input(
    "أدخل Groq API Key:",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "📄 قم برفع ملف PDF",
    type=["pdf"]
)


# =========================================================
# زر إعادة ضبط النظام
# =========================================================

if st.sidebar.button("🗑️ مسح المستند الحالي"):

    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.file_name = None
    st.session_state.processed = False

    st.rerun()


# =========================================================
# معالجة ملف PDF
# =========================================================

if uploaded_file and api_key:

    # التحقق هل الملف تغير
    if (
        st.session_state.file_name != uploaded_file.name
        or st.session_state.vectorstore is None
    ):

        st.session_state.file_name = uploaded_file.name
        st.session_state.processed = False

        # -----------------------------------------------------
        # حفظ الملف مؤقتاً
        # -----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name


        # -----------------------------------------------------
        # تحميل PDF
        # -----------------------------------------------------

        with st.spinner("📖 جاري قراءة ملف PDF..."):

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()


        st.info(
            f"📄 عدد الصفحات: {len(docs)}"
        )


        # -----------------------------------------------------
        # تقسيم النص
        # -----------------------------------------------------

        with st.spinner("✂️ جاري تقسيم النص..."):

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            splits = text_splitter.split_documents(docs)


        st.info(
            f"📝 تم إنشاء {len(splits)} جزء نصي"
        )


        # -----------------------------------------------------
        # Embeddings
        # -----------------------------------------------------

        with st.spinner("🧠 جاري إنشاء Embeddings..."):

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )


        # -----------------------------------------------------
        # ChromaDB
        # -----------------------------------------------------

        with st.spinner("💾 جاري إنشاء قاعدة البيانات المتجهية..."):

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore


        # -----------------------------------------------------
        # Retriever
        # -----------------------------------------------------

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4
            }
        )


        # -----------------------------------------------------
        # Groq LLM
        # -----------------------------------------------------

        llm = ChatGroq(
            groq_api_key=api_key,
            model="openai/gpt-oss-120b",
            temperature=0
        )


        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        system_prompt = """
أنت مساعد ذكي متخصص في تحليل المستندات.

مهمتك هي الإجابة عن أسئلة المستخدم اعتماداً على المعلومات
الموجودة في المستند فقط.

القواعد:

1. لا تخترع معلومات غير موجودة في المستند.
2. إذا لم تجد الإجابة، قل:
   "لا توجد معلومات كافية في المستند للإجابة عن هذا السؤال."
3. إذا كان السؤال باللغة العربية فأجب باللغة العربية.
4. قدم إجابة واضحة ومباشرة.
5. عند وجود أرقام أو تواريخ أو أسماء في المستند، حاول نقلها بدقة.
6. يمكنك تلخيص المعلومات الموجودة في المستند.
7. يمكنك المقارنة بين المعلومات إذا كانت موجودة في المستند.

السياق المسترجع من المستند:

{context}
"""


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )


        # -----------------------------------------------------
        # إنشاء Question Answer Chain
        # -----------------------------------------------------

        question_answer_chain = create_stuff_documents_chain(
            llm,
            prompt
        )


        # -----------------------------------------------------
        # إنشاء RAG Chain
        # -----------------------------------------------------

        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )


        st.session_state.rag_chain = rag_chain
        st.session_state.processed = True


        # حذف الملف المؤقت
        try:
            os.remove(pdf_path)
        except:
            pass


        st.success("✅ تم تحليل المستند وتجهيز نظام RAG!")


# =========================================================
# واجهة المحادثة
# =========================================================

if st.session_state.rag_chain is not None:

    st.divider()

    st.subheader(
        f"💬 اسأل عن: {st.session_state.file_name}"
    )


    user_query = st.text_input(
        "اكتب سؤالك هنا:",
        placeholder="مثال: ما موضوع هذا المستند؟"
    )


    if user_query:

        with st.spinner("🔎 جاري البحث في المستند..."):

            try:

                response = st.session_state.rag_chain.invoke(
                    {
                        "input": user_query
                    }
                )


                st.markdown("### 🤖 الإجابة")

                st.write(
                    response["answer"]
                )


                # -------------------------------------------------
                # عرض المصادر
                # -------------------------------------------------

                if "context" in response:

                    with st.expander("📚 عرض الأجزاء المستخدمة من المستند"):

                        for i, doc in enumerate(
                            response["context"],
                            start=1
                        ):

                            st.markdown(
                                f"**المصدر {i}**"
                            )

                            st.write(
                                doc.page_content[:1000]
                            )

                            if "page" in doc.metadata:

                                st.caption(
                                    f"الصفحة: {doc.metadata['page'] + 1}"
                                )

                            st.divider()


            except Exception as e:

                st.error(
                    "❌ حدث خطأ أثناء معالجة السؤال."
                )

                st.code(
                    str(e)
                )


# =========================================================
# تعليمات البداية
# =========================================================

elif not uploaded_file:

    st.info(
        "👈 ارفع ملف PDF من القائمة الجانبية للبدء."
    )

elif not api_key:

    st.warning(
        "⚠️ أدخل Groq API Key أولاً."
    )