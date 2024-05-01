"use strict";
import question from "../data/faq.json" assert {type: "json"};

// FAQ
const faqItems = document.getElementById("faq-list");

function renderQuestions(data) {
    return data.map(values => `<div class="faq-item mb-3" data-aos="fade-down" data-aos-duration="1000">
        <a class="btn" data-bs-toggle="collapse" href="#collapseExample${values.id}" role="button"
            aria-expanded="false" aria-controls="collapseExample${values.id}">
            <i class="fa-solid fa-leaf"></i>${values.question}<i
                class="bi bi-chevron-down fw-bold"></i>
        </a>
        <div class="collapse" id="collapseExample${values.id}">
            <div class="text-accordion">${values.answer}</div>
        </div>
    </div>`).join('');
}

faqItems.innerHTML = renderQuestions(question);


